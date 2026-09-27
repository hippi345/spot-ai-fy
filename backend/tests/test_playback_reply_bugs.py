from __future__ import annotations

import json

from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.chat_messages import prepare_user_visible_reply
from spot_backend.now_playing import _pick_image_url
from spot_backend.playback_reply import format_now_playing_chat_reply


class _Runner:
    conversation_id = "t"

    def run(self, name: str, args: dict) -> str:
        if name != "spotify_playback_state":
            raise AssertionError(name)
        return json.dumps(
            {
                "is_playing": True,
                "item": {
                    "name": "Gravity",
                    "artists": [{"name": "John Mayer"}],
                },
            }
        )


def test_whats_playing_reply_does_not_double_artist() -> None:
    outcome = try_deterministic_chat_reply("What's playing?", _Runner())
    assert outcome is not None
    assert outcome.reply == "You're listening to Gravity by John Mayer."
    assert "Mayer Mayer" not in outcome.reply


def test_ground_reply_does_not_turn_john_mayer_into_doubled_name() -> None:
    tool_json = json.dumps(
        {
            "item": {
                "name": "Gravity",
                "artists": [{"name": "John Mayer"}],
            }
        }
    )
    reply = '"Gravity" by John Mayer is on right now.'
    fixed = prepare_user_visible_reply(reply, [tool_json])
    assert "Mayer Mayer" not in fixed
    assert "John Mayer" in fixed


def test_pick_image_url_skips_empty_and_zero_width() -> None:
    images = [
        {"url": "", "width": 640},
        {"url": "https://cdn.test/ok.jpg", "width": 0, "height": 64},
        {"url": "https://cdn.test/large.jpg", "width": 640},
    ]
    small = _pick_image_url(images, prefer_largest=False, target_px=64)
    assert small == "https://cdn.test/ok.jpg"
    large = _pick_image_url(images, prefer_largest=True)
    assert large == "https://cdn.test/large.jpg"


def test_format_now_playing_strips_redundant_by_suffix() -> None:
    player = {
        "item": {
            "name": "Gravity by John Mayer",
            "artists": [{"name": "John Mayer"}],
        }
    }
    assert format_now_playing_chat_reply(player) == "You're listening to Gravity by John Mayer."
