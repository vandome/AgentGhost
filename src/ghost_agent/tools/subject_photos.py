"""Reference photos for `image_generation(subjects=[...])` (§4MG).

The image model draws a likeness from PIXELS, never from a name
([[likeness-comes-from-pixels]] §4JX). Live 2026-10-07: asked for two named
politicians, the agent's own thinking said "let me first find/download
photos of both" and its next action was a text-only render — two strangers.
On the follow-up it guessed Wikimedia URLs (400 ×2), screenshotted the
article pages, and finally passed ONE portrait because the node takes one
reference — the second man was invented again.

So acquisition is the tool's job, not the model's: the model names the
specific real people / places / products in `subjects`, and this module

  1. finds each one's lead image on Wikipedia (one API call over Tor —
     the article's own `pageimages` thumbnail, never a guessed URL),
  2. downloads it into the sandbox's ``subject_photos/`` folder as
     ``ref_<slug>.<ext>`` with a provenance record beside it (reused only
     when that record names the same subject and the file decodes), and
  3. joins several photos side by side into ONE image padded to the render's
     shape — the node accepts exactly one reference (§4JW: a second OOMs) and
     stretches it to the output size without keeping its aspect
     (`fit_reference`), so an unpadded strip of three portraits squashed
     every face.

A subject with no findable photo is an ERROR naming it: drawing a stranger
under a real name is the failure this exists to stop. A search that could not
REACH the encyclopedia says so separately — it is worth one retry, and must
not send the user off to find a photo of someone famous.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from ..utils.logging import Icons, pretty_log

#: Photos joined into one reference. Beyond three the faces get too small
#: in the node's ~0.4 MP reference latents to carry a likeness.
MAX_SUBJECTS = 3
#: Height every photo is scaled to before joining.
COMBINED_HEIGHT = 1024
#: White gap between joined photos, so the model sees separate pictures.
#: §4OZ (2026-10-14, measured on 10 ordinary two-person scenes, same seed per
#: pair): a SEAMLESS composite (head-and-shoulders crops touching, feathered
#: seam, blurred margins) came out as two panels 7/10 vs 5/10 for this strip —
#: an edit preserves what looks like a finished picture; the frame is what tells
#: the model these are separate photos to recompose. Do not rebuild it.
COMBINED_GAP = 16
#: The render's shape when the caller asked for none and several subjects
#: are joined — the node's own default (768x512).
DEFAULT_ASPECT = 1.5
#: Sandbox folder for the fetched photos: never a user's own file name.
PHOTO_DIR = "subject_photos"
LOOKUP_TIMEOUT_S = 20
LOOKUP_ATTEMPTS = 2
#: Thumbnail width asked of the API (Wikimedia serves its standard steps;
#: a free-form width 400s from upload.wikimedia.org — §4JX).
THUMB_WIDTH = 1024

_STOP = {"the", "a", "an", "of", "in", "at", "on", "and", "de", "la", "le",
         "ο", "η", "το", "οι", "του", "της"}
_GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")


class SearchUnavailable(LookupError):
    """The encyclopedia could not be reached — NOT "no photo exists"."""


class PhotoUnavailable(LookupError):
    """The article EXISTS but its photo could not be fetched right now (the
    download failed, or what arrived is not a complete image) — NOT "no
    photo exists" either (§4PB: a 503 on the thumbnail used to tell the
    user to upload a photo of a famous person)."""

    def __init__(self, reason: str, page: str = "", title: str = ""):
        super().__init__(reason)
        self.page, self.title = page, title


#: §4PB: one token of a multi-word name may differ by ONE edit from the
#: title's token ("Zoi" ↔ "Zoe", a transliteration) — but only when another
#: token of at least this length matched exactly (the surname anchors it;
#: "Obama" alone must never match "Osama").
FUZZY_ANCHOR_LEN = 4


def _fold(text: str) -> str:
    """Lower-case, accents stripped — "Σαμαράς" and "σαμαρας" compare equal."""
    nfkd = unicodedata.normalize("NFKD", str(text or "").lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", _fold(text), flags=re.UNICODE)
            if len(t) >= 2 and t not in _STOP]


def _one_edit_apart(a: str, b: str) -> bool:
    """Levenshtein distance ≤ 1 (one substitution, insertion or deletion)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def title_matches(subject: str, title: str) -> bool:
    """The article is ABOUT the subject: EVERY content word of the name is
    in the title ("Tsipras" → "Alexis Tsipras", "Toyota Corolla" → "Toyota
    Corolla (E210)"). The reverse direction (title words ⊆ name) was dropped
    in review: "Alexis Tsipras and Antonis Samaras" matched "Alexis Tsipras"
    and rendered ONE man's photo as both, "Michael Jordan" matched "Jordan".
    A search for "Antonis Samaras" also returns "New Democracy (Greece)"
    with the party flag as its image — that must not become his face.
    §4PB: ONE token may be one edit off ("Zoi" ↔ "Zoe") when another token
    of FUZZY_ANCHOR_LEN+ letters matched exactly; a single-token name needs
    an exact token ("Obama" never matches "Osama bin Laden")."""
    want = _tokens(subject)
    have = set(_tokens(title))
    if not want:
        return False
    missing = [t for t in want if t not in have]
    if not missing:
        return True
    if len(missing) != 1:
        return False
    anchored = any(t in have and len(t) >= FUZZY_ANCHOR_LEN for t in want if t != missing[0])
    return anchored and any(_one_edit_apart(missing[0], h) for h in have)


def title_is_exact(subject: str, title: str) -> bool:
    """The title IS the name (folded, ignoring a parenthetical qualifier)."""
    return _tokens(re.sub(r"\s*\([^)]*\)\s*$", "", title)) == _tokens(subject) and bool(_tokens(subject))


def slug(subject: str) -> str:
    s = re.sub(r"[^\w]+", "_", _fold(subject), flags=re.UNICODE).strip("_")
    return (s or "subject")[:60]


def _languages(subject: str) -> List[str]:
    return ["el", "en"] if _GREEK_RE.search(subject or "") else ["en"]


def lookup_photo_url(subject: str, proxy: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """(image_url, article_title, article_url) for the subject's lead image,
    or None when the encyclopedia answered and nothing matches. Raises
    SearchUnavailable when it could not be asked (network, non-200, bad
    JSON) in any language. Synchronous — run it in a thread."""
    import curl_cffi.requests as creq
    from urllib.parse import quote as _quote
    proxies = {"https": proxy, "http": proxy} if proxy else None
    answered = False
    last_err = ""
    base = {"action": "query", "format": "json", "redirects": 1,
            "prop": "pageimages|pageprops", "piprop": "thumbnail",
            "pithumbsize": THUMB_WIDTH, "ppprop": "disambiguation"}

    def _ask(lang, extra):
        r = creq.get(f"https://{lang}.wikipedia.org/w/api.php", params={**base, **extra},
                     proxies=proxies, timeout=LOOKUP_TIMEOUT_S, impersonate="chrome")
        if r.status_code != 200:
            raise SearchUnavailable(f"HTTP {r.status_code}")
        q = (r.json() or {}).get("query") or {}
        return q.get("pages") or {}, q.get("redirects") or []

    def _url(lang, title):
        return f"https://{lang}.wikipedia.org/wiki/{_quote(title.replace(' ', '_'))}"

    for lang in _languages(subject):
        # §4PB step 1: the name AS a title, redirects followed — "Napoleon
        # Bonaparte" → "Napoleon" is the encyclopedia's own say-so, not a
        # guess by token overlap.
        try:
            pages, redirects = _ask(lang, {"titles": subject})
        except Exception as e:  # noqa: BLE001 — network / JSON / non-200
            last_err = e.__class__.__name__ if not isinstance(e, SearchUnavailable) else str(e)
            continue
        answered = True
        redirected_to = {str(r.get("to") or "") for r in redirects
                         if _fold(str(r.get("from") or "")) == _fold(subject)}
        for page in pages.values():
            title = str(page.get("title") or "")
            if "disambiguation" in (page.get("pageprops") or {}):
                continue
            src = str((page.get("thumbnail") or {}).get("source") or "")   # a missing page has none
            if src and (title_is_exact(subject, title) or title in redirected_to):
                return src, title, _url(lang, title)
        # step 2: search, exact titles before token-superset matches
        try:
            pages, _ = _ask(lang, {"generator": "search", "gsrsearch": subject, "gsrlimit": 5})
        except Exception as e:  # noqa: BLE001
            last_err = e.__class__.__name__ if not isinstance(e, SearchUnavailable) else str(e)
            continue
        hits = []
        for page in sorted(pages.values(), key=lambda p: p.get("index", 99)):
            title = str(page.get("title") or "")
            if "disambiguation" in (page.get("pageprops") or {}):
                continue
            src = str((page.get("thumbnail") or {}).get("source") or "")
            if not src or not title_matches(subject, title):
                continue
            hits.append((0 if title_is_exact(subject, title) else 1, title, src))
        if hits:
            hits.sort(key=lambda h: h[0])
            _, title, src = hits[0]
            return src, title, _url(lang, title)
    if not answered:
        raise SearchUnavailable(f"the encyclopedia could not be reached ({last_err or 'no answer'})")
    return None


def photo_is_complete(path: Path) -> bool:
    """The WHOLE file decodes as an image — a download cut off after its
    first megabyte has a valid header and was reused forever (review)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:  # noqa: BLE001
        return False


def _meta_path(photo: Path) -> Path:
    return photo.with_suffix(photo.suffix + ".json")


def _cached(folder: Path, stem: str, subject: str) -> Optional[Tuple[Path, str]]:
    """A previously fetched photo of THIS subject: its provenance record
    names the same subject, and the file decodes completely. Anything else
    (no record, another person whose name slugs the same, a truncated
    file) is fetched again."""
    for ext in (".jpg", ".png"):
        p = folder / f"{stem}{ext}"
        if not p.is_file():
            continue
        try:
            meta = json.loads(_meta_path(p).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if _fold(meta.get("subject", "")) == _fold(subject) and photo_is_complete(p):
            return p, str(meta.get("page") or "")
    return None


async def fetch_subject_photo(subject: str, sandbox_dir: Path, tor_proxy: Optional[str],
                              *, is_image) -> Tuple[Path, str]:
    """(local photo, its article). Raises LookupError(reason) —
    SearchUnavailable when the encyclopedia could not be asked."""
    folder = Path(sandbox_dir) / PHOTO_DIR
    stem = f"ref_{slug(subject)}"
    hit = await asyncio.to_thread(_cached, folder, stem, subject)
    if hit is not None:
        pretty_log("Subject Photo", f"{subject} → {hit[0].name} ({photo_title(hit[0]) or '?'}, cached)",
                   icon=Icons.TOOL_DOWN)
        return hit
    from ..utils.egress_guard import resolve_egress_proxy
    proxy = resolve_egress_proxy(tor_proxy, "https://en.wikipedia.org/")
    if proxy and proxy.startswith("socks5://"):
        proxy = proxy.replace("socks5://", "socks5h://")
    found = None
    for attempt in range(LOOKUP_ATTEMPTS):
        try:
            found = await asyncio.to_thread(lookup_photo_url, subject, proxy)
            break
        except SearchUnavailable:
            if attempt + 1 >= LOOKUP_ATTEMPTS:
                raise
            # a refused / dead circuit: the next attempt gets a fresh one
            from ..utils.helpers import request_new_tor_identity
            await asyncio.to_thread(request_new_tor_identity)
            await asyncio.sleep(3)
    if not found:
        raise LookupError("no encyclopedia article with a photo matches that name")
    url, title, page = found
    ext = ".png" if url.split("?", 1)[0].lower().endswith(".png") else ".jpg"
    folder.mkdir(parents=True, exist_ok=True)
    # Download under a temporary name and promote only a COMPLETE image: a
    # transfer that failed half-way left a truncated file under the final
    # name (`tool_download_file` writes in place).
    part = f"{PHOTO_DIR}/{stem}.part-{uuid.uuid4().hex[:8]}{ext}"
    from .file_system import tool_download_file
    res = await tool_download_file(url=url, sandbox_dir=Path(sandbox_dir),
                                   tor_proxy=tor_proxy, filename=part)
    tmp = Path(sandbox_dir) / part
    try:
        # §4PB: the article exists — a failed or truncated download is a
        # TRANSIENT fault (PhotoUnavailable), never "no photo of this person"
        if not str(res).startswith("SUCCESS") or not tmp.is_file():
            raise PhotoUnavailable(f"the photo from {page} could not be downloaded", page, title)
        head = await asyncio.to_thread(lambda: tmp.open("rb").read(16))
        if not is_image(head) or not await asyncio.to_thread(photo_is_complete, tmp):
            raise PhotoUnavailable(f"what {page} served is not a complete photo", page, title)
        final = folder / f"{stem}{ext}"
        os.replace(tmp, final)
        _meta_path(final).write_text(json.dumps(
            {"subject": subject, "title": title, "page": page, "url": url}, ensure_ascii=False),
            encoding="utf-8")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    pretty_log("Subject Photo", f"{subject} → {final.name} ({title})", icon=Icons.TOOL_DOWN)
    return final, page


#: Render shapes the node offers lie well inside this band; a wider ask is
#: refused BEFORE a canvas is built (§4PB: width=200000,height=1 spent 14 s
#: allocating before MemoryError).
ASPECT_BAND = (0.25, 4.0)


def photo_title(photo: Path) -> str:
    """The article title recorded beside a fetched photo ("" if unknown)."""
    try:
        return str(json.loads(_meta_path(photo).read_text(encoding="utf-8")).get("title") or "")
    except Exception:  # noqa: BLE001
        return ""


def _pad_to_aspect(im, aspect: float):
    """``im`` centred on a white canvas of width/height == ``aspect``."""
    from PIL import Image
    if not (ASPECT_BAND[0] <= float(aspect) <= ASPECT_BAND[1]):
        raise ValueError(f"aspect {aspect:.3g} outside {ASPECT_BAND[0]}–{ASPECT_BAND[1]}")
    w, h = im.size
    if abs(w / h - aspect) < 0.01:
        return im
    cw, ch = (round(h * aspect), h) if w / h < aspect else (w, round(w / aspect))
    canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
    canvas.paste(im, ((cw - w) // 2, (ch - h) // 2))
    return canvas


def combine_side_by_side(paths: List[Path], aspect: Optional[float] = None) -> bytes:
    """One JPEG with the photos left to right at a common height, padded to
    ``aspect`` (width/height) so the node's stretch-to-size keeps faces
    undistorted. ``aspect=None`` keeps the strip's own shape."""
    from PIL import Image, ImageOps
    ims = []
    for p in paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w = max(1, round(im.width * COMBINED_HEIGHT / im.height))
            cap = round(COMBINED_HEIGHT * 1.5)
            if w <= cap:
                ims.append(ImageOps.fit(im, (w, COMBINED_HEIGHT), Image.Resampling.LANCZOS,
                                        centering=(0.5, 0.3)))
            else:
                # a panorama would squeeze the others into slivers — but
                # CROPPING it cut people out of "the scene to keep" (§4PB):
                # scale it to the cap and letterbox the rest
                h = max(1, round(im.height * cap / im.width))
                panel = Image.new("RGB", (cap, COMBINED_HEIGHT), (255, 255, 255))
                panel.paste(im.resize((cap, h), Image.Resampling.LANCZOS), (0, (COMBINED_HEIGHT - h) // 2))
                ims.append(panel)
    width = sum(i.width for i in ims) + COMBINED_GAP * (len(ims) - 1)
    out = Image.new("RGB", (width, COMBINED_HEIGHT), (255, 255, 255))
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width + COMBINED_GAP
    if aspect:
        out = _pad_to_aspect(out, aspect)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    return buf.getvalue()




def pad_photo(path: Path, aspect: float) -> bytes:
    """One photo padded to ``aspect`` (a requested render shape)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = _pad_to_aspect(ImageOps.exif_transpose(im).convert("RGB"), aspect)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def normalise_subjects(subjects) -> List[str]:
    """Distinct names (by slug, so "Alexis-Tsipras" and "Alexis Tsipras" are
    one) from a list, a JSON list string, or one string of names separated by
    commas, semicolons, new lines or "and"/"&"/"και". Whitespace inside a name
    is collapsed (a new line broke the result record's parse)."""
    if subjects is None:
        return []
    if isinstance(subjects, str):
        txt = subjects.strip()
        parsed = None
        if txt.startswith("["):
            try:
                parsed = json.loads(txt)
            except Exception:  # noqa: BLE001
                parsed = None
        subjects = parsed if isinstance(parsed, list) else \
            re.split(r"\s*(?:,|;|\n|&|\band\b|\bκαι\b)\s*", txt, flags=re.IGNORECASE)
    if not isinstance(subjects, (list, tuple)):
        return []
    out: List[str] = []
    seen = set()
    for s in subjects:
        s = re.sub(r"\s+", " ", str(s or "")).strip()[:120]
        if s and slug(s) not in seen:
            seen.add(slug(s))
            out.append(s)
    return out
