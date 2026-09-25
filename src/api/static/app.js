const elements = {
  stateDot: document.querySelector("#state-dot"),
  systemState: document.querySelector("#system-state"),
  feedBadge: document.querySelector("#feed-badge"),
  videoFeed: document.querySelector("#video-feed"),
  videoStatus: document.querySelector("#video-status"),
  start: document.querySelector("#start-monitoring"),
  stop: document.querySelector("#stop-monitoring"),
  cameraStatus: document.querySelector("#camera-status"),
  aiStatus: document.querySelector("#ai-status"),
  webhookStatus: document.querySelector("#webhook-status"),
  error: document.querySelector("#error-message"),
  alertPanel: document.querySelector("#alert-panel"),
  countdown: document.querySelector("#countdown"),
  alertReason: document.querySelector("#alert-reason"),
  cancel: document.querySelector("#cancel-alert"),
  reviewLabel: document.querySelector("#review-label"),
  latestReview: document.querySelector("#latest-review"),
  eventsList: document.querySelector("#events-list"),
  eventCount: document.querySelector("#event-count"),
  pollStatus: document.querySelector("#poll-status"),
};

let monitorState = null;
let requestInFlight = false;
let statePollInFlight = false;
let eventsPollInFlight = false;
let lastCameraStatus = "off";
let audioContext = null;
let alarmTimer = null;
let clientError = null;

function titleCase(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Keep the status-only message when an invalid response body is returned.
    }
    throw new Error(detail);
  }
  return response.json();
}

async function initializeAudio() {
  if (audioContext) {
    await audioContext.resume();
    return;
  }
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) {
    throw new Error("This browser cannot enable the countdown sound.");
  }
  audioContext = new AudioContext();
  await audioContext.resume();
}

function soundAlarm() {
  if (!audioContext || audioContext.state !== "running") return;
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.type = "square";
  oscillator.frequency.value = 740;
  gain.gain.setValueAtTime(0.0001, audioContext.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.16, audioContext.currentTime + 0.015);
  gain.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + 0.22);
  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start();
  oscillator.stop(audioContext.currentTime + 0.23);
}

function startAlarm() {
  if (alarmTimer || !audioContext) return;
  soundAlarm();
  alarmTimer = window.setInterval(soundAlarm, 850);
}

function stopAlarm() {
  if (alarmTimer) {
    window.clearInterval(alarmTimer);
    alarmTimer = null;
  }
}

function setBusy(value) {
  requestInFlight = value;
  renderState();
}

function refreshVideo() {
  elements.videoFeed.src = `/video.mjpg?t=${Date.now()}`;
}

function renderReview(review) {
  elements.latestReview.replaceChildren();
  if (!review) {
    elements.reviewLabel.textContent = "No review yet";
    elements.latestReview.className = "empty-state";
    elements.latestReview.textContent = "Candidate sequences will appear here after local motion screening.";
    return;
  }

  elements.reviewLabel.textContent = titleCase(review.classification);
  elements.latestReview.className = "review-details";
  const fields = [
    ["Classification", titleCase(review.classification)],
    ["Confidence", `${Math.round(review.confidence * 100)}%`],
    ["Person visible", review.person_visible ? "Yes" : "No"],
    ["Reason", review.reason],
  ];
  fields.forEach(([label, value], index) => {
    const block = document.createElement("div");
    if (index === fields.length - 1) block.className = "review-reason";
    const labelElement = document.createElement("span");
    labelElement.textContent = label;
    const valueElement = document.createElement("strong");
    valueElement.textContent = value;
    block.append(labelElement, valueElement);
    elements.latestReview.append(block);
  });
}

function renderState() {
  if (!monitorState) {
    elements.start.disabled = requestInFlight;
    elements.stop.disabled = true;
    return;
  }

  const state = monitorState.state;
  const active = !["stopped", "camera_error"].includes(state);
  const alerting = ["countdown", "sending"].includes(state);

  elements.systemState.textContent = titleCase(state);
  elements.stateDot.className = `state-dot${alerting ? " alert" : active ? " active" : ""}`;
  elements.start.disabled = requestInFlight || active;
  elements.stop.disabled = requestInFlight || !active || alerting;
  elements.cancel.disabled = requestInFlight || state !== "countdown";

  elements.cameraStatus.textContent = titleCase(monitorState.camera_status);
  elements.aiStatus.textContent = state === "reviewing"
    ? "Reviewing candidate"
    : state === "ai_error"
      ? "Review failed"
      : monitorState.latest_review
        ? titleCase(monitorState.latest_review.classification)
        : "Idle";
  elements.webhookStatus.textContent = titleCase(monitorState.webhook_status);

  const cameraRunning = monitorState.camera_status === "running";
  elements.feedBadge.textContent = cameraRunning ? "Local feed active" : titleCase(monitorState.camera_status);
  elements.videoStatus.textContent = cameraRunning
    ? state === "reviewing" ? "AI reviewing selected candidate frames" : "Local screening active"
    : state === "camera_error" ? "Camera error — check the configured device" : "Start monitoring to enable the camera";

  if (lastCameraStatus !== monitorState.camera_status) {
    lastCameraStatus = monitorState.camera_status;
    refreshVideo();
  }

  const errorMessages = [clientError, monitorState.error, monitorState.persistence_warning].filter(Boolean);
  elements.error.hidden = errorMessages.length === 0;
  elements.error.textContent = errorMessages.join(" ");

  const countdownActive = state === "countdown" && monitorState.active_event_id;
  elements.alertPanel.hidden = !countdownActive;
  if (countdownActive) {
    elements.alertReason.textContent = monitorState.latest_review?.reason
      || "Review the camera status. Cancel only if help is not needed.";
    startAlarm();
  } else {
    stopAlarm();
  }

  renderReview(monitorState.latest_review);
}

function renderCountdown() {
  if (!monitorState || monitorState.state !== "countdown" || !monitorState.alert_deadline) return;
  const remaining = Math.max(0, Math.ceil((Date.parse(monitorState.alert_deadline) - Date.now()) / 1000));
  elements.countdown.textContent = String(remaining);
}

function renderEvents(events) {
  elements.eventsList.replaceChildren();
  elements.eventCount.textContent = `${events.length} ${events.length === 1 ? "event" : "events"}`;
  if (events.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No fall events recorded.";
    elements.eventsList.append(empty);
    return;
  }

  events.forEach((event) => {
    const row = document.createElement("article");
    row.className = "event-row";

    const time = document.createElement("time");
    time.className = "event-time";
    time.dateTime = event.detected_at;
    time.textContent = new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(event.detected_at));

    const status = document.createElement("span");
    status.className = `event-status ${event.status}`;
    status.textContent = titleCase(event.status);

    const reason = document.createElement("span");
    reason.className = "event-reason";
    reason.textContent = event.delivery_error || event.review.reason;

    row.append(time, status, reason);
    elements.eventsList.append(row);
  });
}

async function pollState() {
  if (statePollInFlight) return;
  statePollInFlight = true;
  try {
    monitorState = await requestJson("/api/state");
    clientError = null;
    elements.pollStatus.textContent = "Local service connected";
    renderState();
  } catch (error) {
    clientError = `State update failed: ${error.message}`;
    elements.pollStatus.textContent = "Local service unavailable";
    renderState();
  } finally {
    statePollInFlight = false;
  }
}

async function pollEvents() {
  if (eventsPollInFlight) return;
  eventsPollInFlight = true;
  try {
    const events = await requestJson("/api/events");
    renderEvents(events);
  } catch (error) {
    clientError = `Event history update failed: ${error.message}`;
    renderState();
  } finally {
    eventsPollInFlight = false;
  }
}

async function mutate(path) {
  setBusy(true);
  try {
    monitorState = await requestJson(path, { method: "POST", body: "{}" });
    clientError = null;
    renderState();
    await pollEvents();
  } catch (error) {
    clientError = error.message;
    renderState();
  } finally {
    setBusy(false);
  }
}

elements.start.addEventListener("click", async () => {
  try {
    await initializeAudio();
  } catch (error) {
    clientError = error.message;
  }
  refreshVideo();
  await mutate("/api/monitor/start");
});

elements.stop.addEventListener("click", async () => {
  await mutate("/api/monitor/stop");
  refreshVideo();
});

elements.cancel.addEventListener("click", async () => {
  if (!monitorState?.active_event_id) return;
  stopAlarm();
  await mutate(`/api/alerts/${monitorState.active_event_id}/cancel`);
});

window.setInterval(pollState, 500);
window.setInterval(pollEvents, 2000);
window.setInterval(renderCountdown, 250);
pollState();
pollEvents();
