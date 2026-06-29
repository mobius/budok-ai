"""Progressive SVG bracket rendering for tournament analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from yomi_daemon.analysis.tournament import BracketEntrant, SeriesAnalysis, TournamentAnalysis
from yomi_daemon.protocol import ProtocolModel


@dataclass(frozen=True, slots=True)
class BracketFrame(ProtocolModel):
    """One progressive bracket frame."""

    index: int
    label: str
    revealed_series_id: str | None
    svg: str
    completed_series_ids: list[str]


def build_bracket_frames(analysis: TournamentAnalysis) -> list[BracketFrame]:
    """Build a sequence of SVG frames showing the bracket filling in over time."""

    rounds_payload = _load_rounds_payload(Path(analysis.bracket_path))
    series_by_id = {series.series_id: series for series in analysis.series}
    completed_order = _completed_series_in_render_order(analysis, rounds_payload)

    frames: list[BracketFrame] = [
        BracketFrame(
            index=0,
            label="Initial bracket",
            revealed_series_id=None,
            svg=render_bracket_svg(
                analysis,
                rounds_payload=rounds_payload,
                completed_series_ids=set(),
                frame_label="Initial bracket",
            ),
            completed_series_ids=[],
        )
    ]

    revealed: list[str] = []
    for frame_index, series_id in enumerate(completed_order, start=1):
        revealed.append(series_id)
        frames.append(
            BracketFrame(
                index=frame_index,
                label=f"After {series_id}",
                revealed_series_id=series_id,
                svg=render_bracket_svg(
                    analysis,
                    rounds_payload=rounds_payload,
                    completed_series_ids=set(revealed),
                    frame_label=f"After {series_id}",
                ),
                completed_series_ids=list(revealed),
            )
        )

    missing = [series_id for series_id in completed_order if series_id not in series_by_id]
    if missing:
        raise ValueError(f"Bracket series missing from analysis: {missing}")

    return frames


def render_bracket_svg(
    analysis: TournamentAnalysis,
    *,
    rounds_payload: list[list[dict[str, Any]]] | None = None,
    completed_series_ids: set[str] | None = None,
    frame_label: str,
) -> str:
    """Render one SVG bracket frame for a chosen set of completed series."""

    raw_rounds = rounds_payload or _load_rounds_payload(Path(analysis.bracket_path))
    revealed = completed_series_ids or set()
    series_by_id = {series.series_id: series for series in analysis.series}

    layout_rounds = _build_layout_rounds(raw_rounds, series_by_id, revealed)
    width, height, column_xs, centers, series_heights = _compute_layout(layout_rounds)

    title = Path(analysis.tournament_dir).name
    champion_label = (
        _display_entrant(analysis.champion, include_seed=True)
        if analysis.champion is not None
        and analysis.champion.policy_id
        in {
            layout_series.winner.policy_id
            for round_series in layout_rounds
            for layout_series in round_series
            if layout_series.winner is not None
        }
        else "Champion pending"
    )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)} bracket">',
        """
  <defs>
    <filter id="paper-shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="16" stdDeviation="18" flood-opacity="0.09"/>
    </filter>
    <pattern id="paper-grain" width="8" height="8" patternUnits="userSpaceOnUse">
      <path d="M0 0h8M0 4h8" stroke="rgba(17,17,17,0.018)" stroke-width="0.7"/>
    </pattern>
  </defs>
""",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f4efe4"/>',
        f'<rect x="26" y="26" width="{width - 52}" height="{height - 52}" fill="#f7f2e8" '
        'stroke="#111111" stroke-width="1.8" filter="url(#paper-shadow)"/>',
        f'<rect x="26" y="26" width="{width - 52}" height="{height - 52}" fill="url(#paper-grain)" opacity="0.55"/>',
        f'<line x1="62" y1="104" x2="{width - 62}" y2="104" stroke="#111111" stroke-width="1.6"/>',
        f'<line x1="62" y1="{height - 76}" x2="{width - 62}" y2="{height - 76}" stroke="#111111" stroke-width="1.6"/>',
        '<text x="70" y="76" font-family="Times New Roman, Times, serif" font-size="20" '
        'letter-spacing="2.6" text-transform="uppercase" fill="#52524c">Tournament Bracket</text>',
        '<text x="70" y="145" font-family="Times New Roman, Times, serif" font-size="42" '
        'font-weight="600" fill="#111111">{}</text>'.format(escape(title)),
        f'<text x="{width - 70}" y="76" text-anchor="end" font-family="Times New Roman, Times, serif" '
        'font-size="19" fill="#111111">{}</text>'.format(escape(frame_label)),
        f'<text x="{width - 70}" y="145" text-anchor="end" font-family="Times New Roman, Times, serif" '
        'font-size="20" fill="#5b5a53">{}</text>'.format(escape(champion_label)),
    ]

    for round_index, round_series in enumerate(layout_rounds):
        parts.append(
            f'<text x="{column_xs[round_index]}" y="206" font-family="Times New Roman, Times, serif" '
            'font-size="18" letter-spacing="2.2" fill="#55554d">{}</text>'.format(
                escape(round_series[0].round_name if round_series else f"Round {round_index + 1}")
            )
        )

    for round_index in range(len(layout_rounds) - 1):
        source_x = column_xs[round_index] + 308
        target_x = column_xs[round_index + 1] - 34
        elbow_x = source_x + (target_x - source_x) * 0.52
        for series_index, source in enumerate(layout_rounds[round_index]):
            target_index = series_index // 2
            target_y = centers[round_index + 1][target_index]
            source_y = centers[round_index][series_index]
            parts.append(
                f'<path d="M {source_x:.1f} {source_y:.1f} H {elbow_x:.1f} V {target_y:.1f} H {target_x:.1f}" '
                'fill="none" stroke="#151515" stroke-width="1.7" stroke-linecap="square"/>'
            )

    for round_index, round_series in enumerate(layout_rounds):
        x = column_xs[round_index]
        for series_index, layout_series in enumerate(round_series):
            center_y = centers[round_index][series_index]
            box_height = series_heights[round_index][series_index]
            y = center_y - box_height / 2
            parts.extend(
                _render_series_block(layout_series, x=x, y=y, width=308, height=box_height)
            )

    parts.append(
        f'<text x="70" y="{height - 42}" font-family="Times New Roman, Times, serif" font-size="17" fill="#5c5b54">'
        "Completed series show per-game character picks and inferred final HP. KO losers are rendered at 0 HP; "
        "winner HP uses the last observed post-decision state from the run artifacts."
        "</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def render_bracket_png(
    analysis: TournamentAnalysis,
    *,
    rounds_payload: list[list[dict[str, Any]]] | None = None,
    completed_series_ids: set[str] | None = None,
    frame_label: str,
    scale: int = 2,
) -> bytes:
    """Render one PNG bracket frame using the same layout as the SVG output."""

    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ModuleNotFoundError as exc:
        raise RuntimeError("PNG export requires the 'pillow' dependency to be installed.") from exc

    raw_rounds = rounds_payload or _load_rounds_payload(Path(analysis.bracket_path))
    revealed = completed_series_ids or set()
    series_by_id = {series.series_id: series for series in analysis.series}

    layout_rounds = _build_layout_rounds(raw_rounds, series_by_id, revealed)
    width, height, column_xs, centers, series_heights = _compute_layout(layout_rounds)
    scaled_width = width * scale
    scaled_height = height * scale

    image = Image.new("RGBA", (scaled_width, scaled_height), "#f4efe4")
    draw = ImageDraw.Draw(image)

    panel_box = (26 * scale, 26 * scale, (width - 26) * scale, (height - 26) * scale)
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rectangle(
        (
            panel_box[0],
            panel_box[1] + 12 * scale,
            panel_box[2],
            panel_box[3] + 12 * scale,
        ),
        fill=(0, 0, 0, 30),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=12 * scale))
    image.alpha_composite(shadow)
    draw = ImageDraw.Draw(image)
    draw.rectangle(panel_box, fill="#f7f2e8", outline="#111111", width=max(2, scale * 2))

    for y in range(26 * scale, (height - 26) * scale, 8 * scale):
        draw.line(
            ((26 * scale), y, (width - 26) * scale, y),
            fill=(17, 17, 17, 6),
            width=1,
        )

    _draw_line(draw, (62, 104), (width - 62, 104), scale=scale, width=1.6, fill="#111111")
    _draw_line(
        draw,
        (62, height - 76),
        (width - 62, height - 76),
        scale=scale,
        width=1.6,
        fill="#111111",
    )

    title = Path(analysis.tournament_dir).name
    champion_label = (
        _display_entrant(analysis.champion, include_seed=True)
        if analysis.champion is not None
        and analysis.champion.policy_id
        in {
            layout_series.winner.policy_id
            for round_series in layout_rounds
            for layout_series in round_series
            if layout_series.winner is not None
        }
        else "Champion pending"
    )

    _draw_text(
        draw,
        (70, 76),
        "Tournament Bracket",
        font=_font(20, scale=scale),
        fill="#52524c",
    )
    _draw_text(
        draw,
        (70, 145),
        title,
        font=_font(42, bold=True, scale=scale),
        fill="#111111",
    )
    _draw_text_right(
        draw,
        (width - 70, 76),
        frame_label,
        font=_font(19, scale=scale),
        fill="#111111",
    )
    _draw_text_right(
        draw,
        (width - 70, 145),
        champion_label,
        font=_font(20, scale=scale),
        fill="#5b5a53",
    )

    for round_index, round_series in enumerate(layout_rounds):
        label = round_series[0].round_name if round_series else f"Round {round_index + 1}"
        _draw_text(
            draw,
            (column_xs[round_index], 206),
            label,
            font=_font(18, scale=scale),
            fill="#55554d",
        )

    for round_index in range(len(layout_rounds) - 1):
        source_x = column_xs[round_index] + 308
        target_x = column_xs[round_index + 1] - 34
        elbow_x = source_x + (target_x - source_x) * 0.52
        for series_index, _source in enumerate(layout_rounds[round_index]):
            target_index = series_index // 2
            target_y = centers[round_index + 1][target_index]
            source_y = centers[round_index][series_index]
            _draw_polyline(
                draw,
                [
                    (source_x, source_y),
                    (elbow_x, source_y),
                    (elbow_x, target_y),
                    (target_x, target_y),
                ],
                scale=scale,
                width=1.7,
                fill="#151515",
            )

    for round_index, round_series in enumerate(layout_rounds):
        x = column_xs[round_index]
        for series_index, layout_series in enumerate(round_series):
            center_y = centers[round_index][series_index]
            box_height = series_heights[round_index][series_index]
            y = center_y - box_height / 2
            _draw_series_block_png(
                draw,
                layout_series,
                x=x,
                y=y,
                width=308,
                height=box_height,
                scale=scale,
            )

    footer = (
        "Completed series show per-game character picks and inferred final HP. "
        "KO losers are rendered at 0 HP; winner HP uses the last observed post-decision state from the run artifacts."
    )
    _draw_text(
        draw,
        (70, height - 42),
        footer,
        font=_font(17, scale=scale),
        fill="#5c5b54",
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_series_block(
    series: _LayoutSeries, *, x: float, y: float, width: float, height: float
) -> list[str]:
    parts: list[str] = [
        f'<line x1="{x}" y1="{y}" x2="{x + width}" y2="{y}" stroke="#111111" stroke-width="1.5"/>',
        f'<line x1="{x}" y1="{y + 58}" x2="{x + width}" y2="{y + 58}" stroke="#111111" stroke-width="0.9" opacity="0.45"/>',
    ]
    if series.completed and series.game_lines:
        divider_y = y + 84
        parts.append(
            f'<line x1="{x}" y1="{divider_y}" x2="{x + width}" y2="{divider_y}" stroke="#111111" stroke-width="0.9" opacity="0.45"/>'
        )
    parts.append(
        f'<text x="{x}" y="{y - 10}" font-family="Times New Roman, Times, serif" font-size="14" '
        'letter-spacing="1.4" fill="#5a5a53">{}</text>'.format(escape(series.series_id))
    )

    top_text_y = y + 25
    bottom_text_y = y + 51
    parts.extend(
        _render_entrant_line(
            series.top_entrant,
            x=x + 3,
            y=top_text_y,
            muted=series.completed
            and series.loser is not None
            and series.top_entrant == series.loser,
            winner=series.completed
            and series.winner is not None
            and series.top_entrant == series.winner,
            strike=series.completed
            and series.loser is not None
            and series.top_entrant == series.loser,
        )
    )
    parts.extend(
        _render_entrant_line(
            series.bottom_entrant,
            x=x + 3,
            y=bottom_text_y,
            muted=series.completed
            and series.loser is not None
            and series.bottom_entrant == series.loser,
            winner=series.completed
            and series.winner is not None
            and series.bottom_entrant == series.winner,
            strike=series.completed
            and series.loser is not None
            and series.bottom_entrant == series.loser,
        )
    )

    score_text = series.score if series.completed else series.placeholder_score
    parts.append(
        f'<text x="{x + width}" y="{y + 38}" text-anchor="end" font-family="Times New Roman, Times, serif" '
        'font-size="26" fill="#111111">{}</text>'.format(escape(score_text))
    )

    if series.completed and series.game_lines:
        start_y = y + 104
        for line_index, line in enumerate(series.game_lines):
            parts.extend(
                _render_game_line(
                    line,
                    x=x + 3,
                    y=start_y + line_index * 18,
                )
            )

    parts.append(
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" stroke="#111111" stroke-width="1.2"/>'
    )
    return parts


def _render_entrant_line(
    entrant: BracketEntrant | None,
    *,
    x: float,
    y: float,
    muted: bool,
    winner: bool,
    strike: bool,
) -> list[str]:
    label = _display_entrant(entrant, include_seed=True) if entrant is not None else "TBD"
    fill = "#111111" if winner else "#4e4d48" if muted else "#161616"
    font_weight = "600" if winner else "400"
    parts = [
        f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" font-size="20" '
        f'font-weight="{font_weight}" fill="{fill}">{escape(label)}</text>'
    ]
    if strike:
        strike_width = min(258, 10 + len(label) * 8.7)
        parts.append(
            f'<line x1="{x}" y1="{y - 6}" x2="{x + strike_width}" y2="{y - 6}" stroke="#111111" stroke-width="1.5"/>'
        )
    return parts


def _render_game_line(line: _GameLine, *, x: float, y: float) -> list[str]:
    left_fill = "#111111" if line.winner_side == "left" else "#68665f"
    right_fill = "#111111" if line.winner_side == "right" else "#68665f"
    parts = [
        f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" font-size="13.6" fill="#55554d">{escape(line.label)}</text>',
        f'<text x="{x + 28}" y="{y}" font-family="Times New Roman, Times, serif" font-size="13.6" fill="{left_fill}">{escape(line.left_text)}</text>',
        f'<text x="{x + 184}" y="{y}" font-family="Times New Roman, Times, serif" font-size="13.6" fill="#55554d">def.</text>',
        f'<text x="{x + 220}" y="{y}" font-family="Times New Roman, Times, serif" font-size="13.6" fill="{right_fill}">{escape(line.right_text)}</text>',
    ]
    return parts


def _draw_series_block_png(
    draw: Any,
    series: _LayoutSeries,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    scale: int,
) -> None:
    _draw_line(draw, (x, y), (x + width, y), scale=scale, width=1.5, fill="#111111")
    _draw_line(
        draw,
        (x, y + 58),
        (x + width, y + 58),
        scale=scale,
        width=0.9,
        fill=(17, 17, 17, 115),
    )
    if series.completed and series.game_lines:
        divider_y = y + 84
        _draw_line(
            draw,
            (x, divider_y),
            (x + width, divider_y),
            scale=scale,
            width=0.9,
            fill=(17, 17, 17, 115),
        )

    _draw_text(
        draw,
        (x, y - 10),
        series.series_id,
        font=_font(14, scale=scale),
        fill="#5a5a53",
    )
    _draw_entrant_line_png(
        draw,
        series.top_entrant,
        x=x + 3,
        y=y + 25,
        muted=series.completed and series.loser is not None and series.top_entrant == series.loser,
        winner=series.completed
        and series.winner is not None
        and series.top_entrant == series.winner,
        strike=series.completed and series.loser is not None and series.top_entrant == series.loser,
        scale=scale,
    )
    _draw_entrant_line_png(
        draw,
        series.bottom_entrant,
        x=x + 3,
        y=y + 51,
        muted=series.completed
        and series.loser is not None
        and series.bottom_entrant == series.loser,
        winner=series.completed
        and series.winner is not None
        and series.bottom_entrant == series.winner,
        strike=series.completed
        and series.loser is not None
        and series.bottom_entrant == series.loser,
        scale=scale,
    )

    score_text = series.score if series.completed else series.placeholder_score
    _draw_text_right(
        draw,
        (x + width, y + 38),
        score_text,
        font=_font(26, scale=scale),
        fill="#111111",
    )

    if series.completed and series.game_lines:
        start_y = y + 104
        for line_index, line in enumerate(series.game_lines):
            _draw_game_line_png(
                draw,
                line,
                x=x + 3,
                y=start_y + line_index * 18,
                scale=scale,
            )

    _draw_line(
        draw,
        (x, y + height),
        (x + width, y + height),
        scale=scale,
        width=1.2,
        fill="#111111",
    )


def _draw_entrant_line_png(
    draw: Any,
    entrant: BracketEntrant | None,
    *,
    x: float,
    y: float,
    muted: bool,
    winner: bool,
    strike: bool,
    scale: int,
) -> None:
    label = _display_entrant(entrant, include_seed=True) if entrant is not None else "TBD"
    fill = "#111111" if winner else "#4e4d48" if muted else "#161616"
    font = _font(20, bold=winner, scale=scale)
    _draw_text(draw, (x, y), label, font=font, fill=fill)
    if strike:
        bbox = draw.textbbox((0, 0), label, font=font.image_font)
        width = min(258 * scale, bbox[2] - bbox[0])
        _draw_line(
            draw,
            (x, y - 6),
            (x + width / scale, y - 6),
            scale=scale,
            width=1.5,
            fill="#111111",
        )


def _draw_game_line_png(draw: Any, line: _GameLine, *, x: float, y: float, scale: int) -> None:
    left_fill = "#111111" if line.winner_side == "left" else "#68665f"
    right_fill = "#111111" if line.winner_side == "right" else "#68665f"
    font = _font(13.6, scale=scale)
    _draw_text(draw, (x, y), line.label, font=font, fill="#55554d")
    _draw_text(draw, (x + 28, y), line.left_text, font=font, fill=left_fill)
    _draw_text(draw, (x + 184, y), "def.", font=font, fill="#55554d")
    _draw_text(draw, (x + 220, y), line.right_text, font=font, fill=right_fill)


def _draw_polyline(
    draw: Any,
    points: list[tuple[float, float]],
    *,
    scale: int,
    width: float,
    fill: str | tuple[int, int, int, int],
) -> None:
    scaled = [(round(x * scale), round(y * scale)) for x, y in points]
    draw.line(scaled, fill=fill, width=max(1, round(width * scale)))


def _draw_line(
    draw: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    scale: int,
    width: float,
    fill: str | tuple[int, int, int, int],
) -> None:
    draw.line(
        [
            (round(start[0] * scale), round(start[1] * scale)),
            (round(end[0] * scale), round(end[1] * scale)),
        ],
        fill=fill,
        width=max(1, round(width * scale)),
    )


def _draw_text(
    draw: Any, position: tuple[float, float], text: str, *, font: Any, fill: str
) -> None:
    draw.text(
        (position[0] * font.scale, (position[1] - font.baseline_offset) * font.scale),
        text,
        font=font.image_font,
        fill=fill,
    )


def _draw_text_right(
    draw: Any,
    position: tuple[float, float],
    text: str,
    *,
    font: Any,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font.image_font)
    width = bbox[2] - bbox[0]
    draw.text(
        (
            position[0] * font.scale - width,
            (position[1] - font.baseline_offset) * font.scale,
        ),
        text,
        font=font.image_font,
        fill=fill,
    )


@dataclass(frozen=True, slots=True)
class _LoadedFont:
    image_font: Any
    baseline_offset: float
    scale: int


@lru_cache(maxsize=16)
def _font(size: float, *, bold: bool = False, scale: int = 2) -> _LoadedFont:
    try:
        from PIL import ImageFont
    except ModuleNotFoundError as exc:
        raise RuntimeError("PNG export requires the 'pillow' dependency to be installed.") from exc

    font_size = max(12, round(size * scale))
    font_path = _font_path(bold=bold)
    image_font = ImageFont.truetype(str(font_path), font_size)
    return _LoadedFont(
        image_font=image_font,
        baseline_offset=size * 0.82,
        scale=scale,
    )


def _font_path(*, bold: bool) -> Path:
    candidates = [
        # macOS bundled Times New Roman
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Times.ttc",
        # Linux common serif fallbacks (Dejavu, Liberation, Free, Nimbus)
        f"/usr/share/fonts/truetype/dejavu/DejaVuSerif{'-Bold' if bold else ''}.ttf",
        f"/usr/share/fonts/truetype/liberation2/LiberationSerif-{'Bold' if bold else 'Regular'}.ttf",
        f"/usr/share/fonts/truetype/liberation/LiberationSerif-{'Bold' if bold else 'Regular'}.ttf",
        f"/usr/share/fonts/truetype/freefont/FreeSerif{'Bold' if bold else ''}.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find a Times New Roman or compatible serif font on this system."
    )


def _compute_layout(
    rounds: list[list[_LayoutSeries]],
) -> tuple[int, int, list[int], list[list[float]], list[list[float]]]:
    round_count = len(rounds)
    first_round_count = len(rounds[0]) if rounds else 0
    top_margin = 258
    bottom_margin = 126
    left_margin = 88
    column_width = 308
    column_gap = 144
    max_games = max(
        (len(series.game_lines) for round_series in rounds for series in round_series), default=0
    )
    leaf_pitch = 164 + max(0, max_games - 1) * 20

    first_round_centers: list[float] = [
        float(top_margin + 42 + index * leaf_pitch) for index in range(first_round_count)
    ]
    centers: list[list[float]] = [first_round_centers]
    while len(centers[-1]) > 1:
        previous = centers[-1]
        centers.append(
            [(previous[index] + previous[index + 1]) / 2 for index in range(0, len(previous), 2)]
        )

    series_heights: list[list[float]] = []
    for round_series in rounds:
        series_heights.append(
            [
                74
                + (
                    26 + len(series.game_lines) * 18
                    if series.completed and series.game_lines
                    else 0
                )
                for series in round_series
            ]
        )

    width = int(left_margin * 2 + round_count * column_width + max(0, round_count - 1) * column_gap)
    content_height = 0.0
    for round_index, round_series in enumerate(rounds):
        for series_index, _series in enumerate(round_series):
            bottom = (
                centers[round_index][series_index] + series_heights[round_index][series_index] / 2
            )
            content_height = max(content_height, bottom)
    height = int(content_height + bottom_margin)
    column_xs = [left_margin + index * (column_width + column_gap) for index in range(round_count)]

    return width, height, column_xs, centers, series_heights


def _build_layout_rounds(
    raw_rounds: list[list[dict[str, Any]]],
    series_by_id: dict[str, SeriesAnalysis],
    revealed: set[str],
) -> list[list[_LayoutSeries]]:
    layout_rounds: list[list[_LayoutSeries]] = []
    for round_index, round_payloads in enumerate(raw_rounds):
        layout_round: list[_LayoutSeries] = []
        for series_index, payload in enumerate(round_payloads):
            series_id = _expect_string(payload.get("series_id"), context="series.series_id")
            series = series_by_id[series_id]

            if round_index == 0:
                top_entrant = _parse_entrant(payload.get("high_seed"))
                bottom_entrant = _parse_entrant(payload.get("low_seed"))
            else:
                parent_round = layout_rounds[round_index - 1]
                top_parent = parent_round[series_index * 2]
                bottom_parent = parent_round[series_index * 2 + 1]
                top_entrant = top_parent.winner if top_parent.completed else None
                bottom_entrant = bottom_parent.winner if bottom_parent.completed else None

            completed = (
                series_id in revealed and series.winner is not None and series.loser is not None
            )
            layout_round.append(
                _LayoutSeries(
                    series_id=series_id,
                    round_name=series.round_name,
                    top_entrant=top_entrant,
                    bottom_entrant=bottom_entrant,
                    completed=completed,
                    winner=series.winner if completed else None,
                    loser=series.loser if completed else None,
                    score=f"{series.high_seed_wins}-{series.low_seed_wins}",
                    placeholder_score="",
                    game_lines=_build_game_lines(series) if completed else [],
                )
            )
        layout_rounds.append(layout_round)
    return layout_rounds


def _build_game_lines(series: SeriesAnalysis) -> list[_GameLine]:
    top_policy = series.high_seed.policy_id if series.high_seed is not None else None
    lines: list[_GameLine] = []
    for game in series.games:
        top_participant = game.p1 if top_policy == game.p1.policy_id else game.p2
        bottom_participant = game.p2 if top_participant is game.p1 else game.p1
        winner_side = "left" if top_participant.won else "right"
        lines.append(
            _GameLine(
                label=f"G{game.game_index}",
                left_text=_format_game_side(
                    top_participant, won=winner_side == "left", end_reason=game.end_reason
                ),
                right_text=_format_game_side(
                    bottom_participant, won=winner_side == "right", end_reason=game.end_reason
                ),
                winner_side=winner_side,
            )
        )
    return lines


def _format_game_side(participant: Any, *, won: bool, end_reason: str | None) -> str:
    hp = participant.last_observed_hp
    if end_reason == "ko":
        if won:
            rendered_hp = hp if hp is not None else "?"
        else:
            rendered_hp = 0
    else:
        rendered_hp = hp if hp is not None else "?"
    return f"{_short_policy_name(participant.policy_id)} {_short_character(participant.character)} {rendered_hp}"


def _completed_series_in_render_order(
    analysis: TournamentAnalysis,
    rounds_payload: list[list[dict[str, Any]]],
) -> list[str]:
    position: dict[str, tuple[int, int]] = {}
    for round_index, round_series in enumerate(rounds_payload):
        for series_index, payload in enumerate(round_series):
            series_id = _expect_string(payload.get("series_id"), context="series.series_id")
            position[series_id] = (round_index, series_index)

    completed = [
        series
        for series in analysis.series
        if series.winner is not None and series.loser is not None and series.games
    ]
    completed.sort(
        key=lambda series: (
            max((game.completed_at or "" for game in series.games), default=""),
            position.get(series.series_id, (999, 999)),
        )
    )
    return [series.series_id for series in completed]


def _load_rounds_payload(path: Path) -> list[list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rounds = payload.get("rounds")
    if not isinstance(raw_rounds, list):
        raise TypeError(f"{path} did not contain bracket rounds")
    parsed_rounds: list[list[dict[str, Any]]] = []
    for round_payload in raw_rounds:
        if not isinstance(round_payload, list):
            raise TypeError("Bracket round payload must be a list")
        parsed_rounds.append([item for item in round_payload if isinstance(item, dict)])
    return parsed_rounds


def _display_entrant(entrant: BracketEntrant | None, *, include_seed: bool) -> str:
    if entrant is None:
        return "TBD"
    prefix = f"#{entrant.seed} " if include_seed else ""
    return f"{prefix}{_short_policy_name(entrant.policy_id)}"


def _short_policy_name(policy_id: str | None) -> str:
    if not policy_id:
        return "TBD"
    if "/" not in policy_id:
        return policy_id
    provider, model = policy_id.split("/", 1)
    if provider == "policy":
        return policy_id
    replacements = {
        "google/gemini-3.1-pro-preview": "gemini-3.1-pro",
        "x-ai/grok-4.20-beta": "grok-4.20",
    }
    return replacements.get(policy_id, model)


def _short_character(character: str | None) -> str:
    if not character:
        return "?"
    return character


def _parse_entrant(raw: object) -> BracketEntrant | None:
    if not isinstance(raw, dict):
        return None
    payload = cast(dict[str, object], raw)
    seed = payload.get("seed")
    policy_id = payload.get("policy_id")
    if isinstance(seed, int) and isinstance(policy_id, str) and seed > 0 and policy_id != "TBD":
        return BracketEntrant(seed=seed, policy_id=policy_id)
    return None


def _expect_string(raw: object, *, context: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{context} must be a non-empty string")
    return raw


@dataclass(frozen=True, slots=True)
class _LayoutSeries:
    series_id: str
    round_name: str
    top_entrant: BracketEntrant | None
    bottom_entrant: BracketEntrant | None
    completed: bool
    winner: BracketEntrant | None
    loser: BracketEntrant | None
    score: str
    placeholder_score: str
    game_lines: list["_GameLine"]


@dataclass(frozen=True, slots=True)
class _GameLine:
    label: str
    left_text: str
    right_text: str
    winner_side: str
