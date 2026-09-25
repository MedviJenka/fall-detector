import base64
from typing import Any

import httpx
from pydantic import AnyHttpUrl

from src.models import DeliveryResult, FallEvent


class WebhookNotifier:
    def __init__(
        self,
        webhook_url: AnyHttpUrl | str | None,
        *,
        client: Any | None = None,
    ) -> None:
        self._webhook_url = str(webhook_url) if webhook_url else None
        self._client = client

    async def send(self, event: FallEvent, snapshot: bytes) -> DeliveryResult:
        if self._webhook_url is None:
            return DeliveryResult(
                status="delivery_failed",
                error="No caregiver webhook is configured.",
            )

        payload = {
            "event_id": str(event.event_id),
            "detected_at": event.detected_at.isoformat(),
            "classification": event.review.classification,
            "confidence": event.review.confidence,
            "reason": event.review.reason,
            "needs_help": True,
            "source": "local_webcam",
            "snapshot_data_url": (
                "data:image/jpeg;base64," + base64.b64encode(snapshot).decode("ascii")
            ),
        }
        request = {
            "json": payload,
            "headers": {"X-Fall-Detector-Event-ID": str(event.event_id)},
            "timeout": 5.0,
        }

        try:
            if self._client is not None:
                response = await self._client.post(self._webhook_url, **request)
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(self._webhook_url, **request)
        except httpx.TimeoutException:
            return DeliveryResult(
                status="delivery_failed",
                error="The caregiver webhook timed out.",
            )
        except httpx.HTTPError:
            return DeliveryResult(
                status="delivery_failed",
                error="The caregiver webhook could not be reached.",
            )

        if 200 <= response.status_code < 300:
            return DeliveryResult(status="sent")
        return DeliveryResult(
            status="delivery_failed",
            error=f"The caregiver webhook returned HTTP {response.status_code}.",
        )
