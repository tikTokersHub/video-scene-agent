// ---- Small utilities ----
const escapeHtml = (value) => String(value ?? '').replace(/[<>&"']/g, (char) => ({
  '<': '&lt;',
  '>': '&gt;',
  '&': '&amp;',
  '"': '&quot;',
  "'": '&#39;',
}[char]));

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const heroColumn = document.querySelector('.hero-stagger');
const grid = document.getElementById('evidenceGrid');
const evidenceCount = document.getElementById('evidenceCount');
const selectedEvidencePanel = document.getElementById('selectedEvidencePanel');
const selectedEvidenceImage = document.getElementById('selectedEvidenceImage');
const selectedEvidencePlaceholder = document.getElementById('selectedEvidencePlaceholder');
const selectedEvidenceCaption = document.getElementById('selectedEvidenceCaption');
const selectedEvidenceTimestamp = document.getElementById('selectedEvidenceTimestamp');
const selectedEvidenceFrame = document.getElementById('selectedEvidenceFrame');
const selectedEvidenceSimilarity = document.getElementById('selectedEvidenceSimilarity');
const askInput = document.getElementById('askInput');
const askSend = document.getElementById('askSend');
const answerText = document.querySelector('.answer-text');
const confVal = document.querySelector('.conf-val');
const confFill = document.querySelector('.conf-fill');
const uploadForm = document.getElementById('uploadForm');
const videoFile = document.getElementById('videoFile');
const fileName = document.getElementById('fileName');
const videoIdInput = document.getElementById('videoId');
const sampleEvery = document.getElementById('sampleEvery');
const generateRules = document.getElementById('generateRules');
const normalRules = document.getElementById('normalRules');
const abnormalRules = document.getElementById('abnormalRules');
const ruleContext = document.getElementById('ruleContext');
const rulesPanel = document.getElementById('rulesPanel');
const rulesStatus = document.getElementById('rulesStatus');
const editRulesButton = document.getElementById('editRulesButton');
const normalRulesList = document.getElementById('normalRulesList');
const abnormalRulesList = document.getElementById('abnormalRulesList');
const normalRuleCount = document.getElementById('normalRuleCount');
const abnormalRuleCount = document.getElementById('abnormalRuleCount');
const addNormalRuleButton = document.getElementById('addNormalRuleButton');
const addAbnormalRuleButton = document.getElementById('addAbnormalRuleButton');
const rulesEditor = document.getElementById('rulesEditor');
const normalRuleEdit = document.getElementById('normalRuleEdit');
const abnormalRuleEdit = document.getElementById('abnormalRuleEdit');
const rulesEditorMessage = document.getElementById('rulesEditorMessage');
const cancelRuleEditButton = document.getElementById('cancelRuleEditButton');
const saveRuleEditButton = document.getElementById('saveRuleEditButton');
const prepareBtn = document.getElementById('prepareBtn');
const activeVideoPill = document.getElementById('activeVideoPill');
const uploadResult = document.getElementById('uploadResult');
const processStatus = document.getElementById('processStatus');
const processTitle = document.getElementById('processTitle');
const processDetail = document.getElementById('processDetail');
const processCompleteBadge = document.getElementById('processCompleteBadge');
const processPercent = document.getElementById('processPercent');
const dropzone = document.getElementById('dropzone');
const consoleTitle = document.querySelector('.console-title');
const consoleLive = document.querySelector('.console-live');

let activeVideoId = '';
let renderedEvidenceItems = [];

function formatTimestamp(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return '';
  const total = Math.max(0, Number(seconds));
  const min = Math.floor(total / 60);
  const sec = (total % 60).toFixed(1).padStart(4, '0');
  return `${String(min).padStart(2, '0')}:${sec}`;
}

function frameUrlFromPath(framePath) {
  const normalized = String(framePath || '').replaceAll('\\', '/');
  const match = normalized.match(/data\/uploads\/([^/]+)\/frames\/([^/]+)$/);
  if (!match) return '';
  return `/frames/${encodeURIComponent(match[1])}/${encodeURIComponent(match[2])}`;
}

function formatSimilarity(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
  return `${Math.round(clamp(Number(value), 0, 1) * 100)}%`;
}

function hideSelectedEvidence() {
  if (!selectedEvidencePanel) return;
  selectedEvidencePanel.hidden = true;
  heroColumn?.classList.remove('evidence-open');

  if (selectedEvidenceImage) {
    selectedEvidenceImage.hidden = true;
    selectedEvidenceImage.removeAttribute('src');
  }

  if (selectedEvidencePlaceholder) selectedEvidencePlaceholder.hidden = false;
}

function showSelectedEvidence(item, index) {
  if (!selectedEvidencePanel) return;

  const imageUrl = frameUrlFromPath(item.frame_path);
  const timestamp = item.timestamp_sec !== undefined
    ? formatTimestamp(item.timestamp_sec)
    : item.ts || '--';
  const frameLabel = item.frame_idx !== undefined ? `#${item.frame_idx}` : `#${index + 1}`;

  selectedEvidencePanel.hidden = false;
  heroColumn?.classList.add('evidence-open');
  if (selectedEvidenceCaption) {
    selectedEvidenceCaption.textContent = item.caption || 'No caption returned for this frame.';
  }
  if (selectedEvidenceTimestamp) selectedEvidenceTimestamp.textContent = timestamp;
  if (selectedEvidenceFrame) selectedEvidenceFrame.textContent = frameLabel;
  if (selectedEvidenceSimilarity) {
    selectedEvidenceSimilarity.textContent = formatSimilarity(item.similarity_score);
  }

  if (selectedEvidenceImage && selectedEvidencePlaceholder) {
    if (imageUrl) {
      selectedEvidenceImage.hidden = false;
      selectedEvidencePlaceholder.hidden = true;
      selectedEvidenceImage.src = imageUrl;
      selectedEvidenceImage.onerror = () => {
        selectedEvidenceImage.hidden = true;
        selectedEvidencePlaceholder.hidden = false;
      };
    } else {
      selectedEvidenceImage.hidden = true;
      selectedEvidenceImage.removeAttribute('src');
      selectedEvidencePlaceholder.hidden = false;
    }
  }

  selectedEvidencePanel.scrollIntoView({
    behavior: 'smooth',
    block: 'nearest',
  });
}

function selectFrame(frame) {
  document.querySelectorAll('.frame').forEach((item) => item.classList.remove('active'));
  frame.classList.add('active');
}

function renderFrame(item, index, active = false) {
  const div = document.createElement('div');
  div.className = `frame${active ? ' active' : ''}`;
  div.dataset.evidenceIndex = String(index);

  const imageUrl = frameUrlFromPath(item.frame_path);
  const ts = item.timestamp_sec !== undefined
    ? formatTimestamp(item.timestamp_sec)
    : item.ts || `frame ${item.frame_idx ?? index}`;
  const caption = escapeHtml(item.caption || '');
  const visual = imageUrl
    ? `<img class="frame-img" src="${escapeHtml(imageUrl)}" alt="" loading="lazy" />`
    : '<div class="frame-placeholder">Frame unavailable</div>';

  div.title = caption;
  div.innerHTML = `${visual}
    <div class="frame-corner"></div>
    <div class="frame-ts">${escapeHtml(ts)}</div>`;

  return div;
}

function handleEvidenceGridClick(event) {
  const frame = event.target.closest('.frame');

  if (!frame || !grid.contains(frame)) return;

  const index = Number(frame.dataset.evidenceIndex);
  const item = renderedEvidenceItems[index];

  if (!item) return;

  selectFrame(frame);
  showSelectedEvidence(item, index);

  if (item.caption) {
    const safeCaption = escapeHtml(item.caption);
    answerText.dataset.baseAnswer = answerText.dataset.baseAnswer || answerText.innerHTML;
    answerText.innerHTML = `${answerText.dataset.baseAnswer}<br><br><span class="hl">Selected evidence:</span> ${safeCaption}`;
  }
}

function updateEvidenceCount(count) {
  if (!evidenceCount) return;
  evidenceCount.textContent = `${count} frame${count === 1 ? '' : 's'}`;
}

function renderEvidence(evidenceItems) {
  grid.innerHTML = '';
  answerText.dataset.baseAnswer = '';
  const items = Array.isArray(evidenceItems) ? evidenceItems : [];
  renderedEvidenceItems = items.slice(0, 8);
  hideSelectedEvidence();
  updateEvidenceCount(items.length);

  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'evidence-empty';
    empty.textContent = 'Ask a question to retrieve evidence frames.';
    grid.appendChild(empty);
    return;
  }

  renderedEvidenceItems.forEach((item, index) => {
    grid.appendChild(renderFrame(item, index, false));
  });
}

renderEvidence([]);

function updateActiveVideo(videoId, fileLabel) {
  activeVideoId = videoId || '';
  activeVideoPill.textContent = activeVideoId ? `Active: ${activeVideoId}` : 'No active video';
  activeVideoPill.classList.toggle('ready', Boolean(activeVideoId));
  setRuleActionsEnabled(Boolean(activeVideoId));

  if (consoleTitle && fileLabel) {
    consoleTitle.textContent = `scene_agent / ${fileLabel}`;
  }
}

function pluralizeRule(count) {
  return `${count} rule${count === 1 ? '' : 's'}`;
}

function setRuleActionsEnabled(enabled) {
  [editRulesButton, addNormalRuleButton, addAbnormalRuleButton].forEach((button) => {
    if (button) button.disabled = !enabled;
  });
}

function setRulesStatus(message, isError = false) {
  if (!rulesStatus) return;
  rulesStatus.textContent = message;
  rulesStatus.classList.toggle('error', isError);
}

function renderRuleItems(container, rules, type, emptyLabel) {
  if (!container) return;
  const items = Array.isArray(rules) ? rules : [];

  if (!items.length) {
    container.innerHTML = `<div class="rule-item empty">${escapeHtml(emptyLabel)}</div>`;
    return;
  }

  container.innerHTML = items.map((item) => {
    const rule = typeof item === 'string' ? item : item.rule;
    const ruleId = typeof item === 'string' ? '' : item.id;
    const canDelete = Boolean(ruleId && activeVideoId);

    return `<div class="rule-item">
      <span class="rule-bullet ${type}"></span>
      <span class="rule-text">${escapeHtml(rule)}</span>
      ${canDelete ? `<button class="rule-delete" type="button" data-delete-rule-id="${escapeHtml(ruleId)}">Delete</button>` : ''}
    </div>`;
  }).join('');
}

function renderAppliedRules(data) {
  const normal = data?.normal_rules || [];
  const abnormal = data?.abnormal_rules || [];
  const total = Number(data?.total_rules ?? (normal.length + abnormal.length));
  const hasVideo = Boolean(data?.video_id);

  renderRuleItems(normalRulesList, normal, 'normal', 'No normal rules applied yet.');
  renderRuleItems(abnormalRulesList, abnormal, 'abnormal', 'No abnormal rules applied yet.');

  if (normalRuleCount) normalRuleCount.textContent = pluralizeRule(normal.length);
  if (abnormalRuleCount) abnormalRuleCount.textContent = pluralizeRule(abnormal.length);

  if (hasVideo) {
    setRulesStatus(total
      ? `${pluralizeRule(total)} loaded for ${data.video_id}`
      : `No rules found for ${data.video_id}`);
  } else {
    setRulesStatus('Upload a video to load applied rules.');
  }

  setRuleActionsEnabled(Boolean(activeVideoId));
}

async function fetchVideoRules(videoId) {
  const response = await fetch(`/api/videos/${encodeURIComponent(videoId)}/rules`, {
    cache: 'no-store',
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function loadVideoRules(videoId) {
  if (!videoId) {
    renderAppliedRules(null);
    return null;
  }

  setRulesStatus(`Loading rules for ${videoId}...`);

  try {
    const data = await fetchVideoRules(videoId);
    renderAppliedRules(data);
    return data;
  } catch (error) {
    setRulesStatus(`Could not load rules: ${error.message}`, true);
    renderRuleItems(normalRulesList, [], 'normal', 'Rules could not be loaded.');
    renderRuleItems(abnormalRulesList, [], 'abnormal', 'Rules could not be loaded.');
    console.error(error);
    return null;
  }
}

function openRulesEditor(type = 'normal') {
  if (!activeVideoId || !rulesEditor) return;
  rulesEditor.hidden = false;
  rulesEditorMessage.textContent = `Adding rules to ${activeVideoId}`;
  const target = type === 'abnormal' ? abnormalRuleEdit : normalRuleEdit;
  setTimeout(() => target?.focus(), 0);
}

function closeRulesEditor() {
  if (!rulesEditor) return;
  rulesEditor.hidden = true;
  normalRuleEdit.value = '';
  abnormalRuleEdit.value = '';
  rulesEditorMessage.textContent = '';
}

function parseRuleEditorValue(value) {
  return String(value || '')
    .split(/[\n;]+/)
    .map((rule) => rule.trim().replace(/^[-*]\s*/, ''))
    .filter(Boolean);
}

async function saveRuleEdits(event) {
  event.preventDefault();

  if (!activeVideoId) return;

  const normal = parseRuleEditorValue(normalRuleEdit.value);
  const abnormal = parseRuleEditorValue(abnormalRuleEdit.value);

  if (!normal.length && !abnormal.length) {
    rulesEditorMessage.textContent = 'Add at least one rule.';
    return;
  }

  saveRuleEditButton.disabled = true;
  rulesEditorMessage.textContent = 'Saving rules...';

  try {
    const response = await fetch(`/api/videos/${encodeURIComponent(activeVideoId)}/rules`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        normal_rules: normal,
        abnormal_rules: abnormal,
      }),
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const data = await response.json();
    closeRulesEditor();
    renderAppliedRules(data);
  } catch (error) {
    rulesEditorMessage.textContent = error.message;
    console.error(error);
  } finally {
    saveRuleEditButton.disabled = false;
  }
}

async function deleteRule(ruleId, button) {
  if (!activeVideoId || !ruleId) return;

  button.disabled = true;
  button.textContent = 'Deleting';
  setRulesStatus('Deleting rule...');

  try {
    const response = await fetch(
      `/api/videos/${encodeURIComponent(activeVideoId)}/rules/${encodeURIComponent(ruleId)}`,
      { method: 'DELETE' }
    );

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const data = await response.json();
    renderAppliedRules(data);
  } catch (error) {
    button.disabled = false;
    button.textContent = 'Delete';
    setRulesStatus(`Could not delete rule: ${error.message}`, true);
    console.error(error);
  }
}

function handleRulePanelClick(event) {
  const deleteButton = event.target.closest('[data-delete-rule-id]');

  if (!deleteButton) return;

  deleteRule(deleteButton.dataset.deleteRuleId, deleteButton);
}

function setConsoleMode(label) {
  if (!consoleLive) return;
  consoleLive.classList.toggle('idle', label === 'IDLE');
  consoleLive.innerHTML = `<span class="rec-dot"></span> ${escapeHtml(label)}`;
}

const stageLabels = {
  upload: 'Uploading video',
  queued: 'Waiting to prepare',
  copy: 'Storing video',
  extract: 'Extracting frames',
  caption: 'Captioning frames',
  ingest: 'Indexing scenes',
  rules: 'Preparing rules',
  complete: 'Ready',
  failed: 'Failed',
};

const stageIndexes = {
  upload: 0,
  queued: 0,
  copy: 0,
  extract: 1,
  caption: 2,
  ingest: 3,
  rules: 4,
  complete: 5,
  failed: 0,
};

const sleep = (ms) => new Promise((resolve) => {
  setTimeout(resolve, ms);
});

function setProcessStep(stage) {
  const activeIndex = stageIndexes[stage] ?? 0;
  processStatus.querySelectorAll('.process-steps span').forEach((step, index) => {
    const isComplete = stage === 'complete';
    step.classList.toggle('done', isComplete || index < activeIndex);
    step.classList.toggle('active', !isComplete && index === activeIndex);
  });
}

function renderPrepareProgress(job) {
  const stage = job.stage || 'queued';
  const progress = typeof job.progress === 'number' ? clamp(job.progress, 0, 100) : null;
  const stageProgress = typeof job.stage_progress === 'number'
    ? clamp(job.stage_progress, 0, 100)
    : null;
  const isCaptioning = stage === 'caption';
  const isComplete = job.status === 'complete';
  const shownPercent = isCaptioning && stageProgress !== null ? stageProgress : progress;

  processStatus.hidden = false;
  processStatus.classList.add('is-active');
  processStatus.classList.toggle('complete', isComplete);
  processStatus.classList.toggle('failed', job.status === 'failed');
  processTitle.textContent = stageLabels[stage] || 'Preparing video';
  processDetail.textContent = job.message || 'Preparing video';
  processCompleteBadge.hidden = !isComplete;
  processPercent.hidden = shownPercent === null;
  processPercent.textContent = isComplete
    ? '100%'
    : isCaptioning && job.current && job.total
    ? `${Math.round(shownPercent)}%`
    : shownPercent !== null ? `${Math.round(shownPercent)}%` : '';
  setProcessStep(stage);
  setConsoleMode(job.status === 'failed' ? 'ERROR' : job.status === 'complete' ? 'READY' : 'PROCESSING');
}

function startProcess(title, detail = 'Uploading file') {
  processStatus.hidden = false;
  processStatus.classList.add('is-active');
  processStatus.classList.remove('complete', 'failed');
  processTitle.textContent = title;
  processDetail.textContent = detail;
  processCompleteBadge.hidden = true;
  processPercent.hidden = false;
  processPercent.textContent = '0%';
  setProcessStep('upload');
  setConsoleMode('PROCESSING');
}

function stopProcess(success, detail, result = null) {
  processStatus.classList.remove('is-active');
  processStatus.classList.toggle('complete', success);
  processStatus.classList.toggle('failed', !success);
  processTitle.textContent = success ? 'Ready' : 'Stopped';
  const frames = Number(result?.frames_extracted || 0);
  processDetail.textContent = success && frames
    ? `${frames.toLocaleString()} frames captioned and indexed`
    : detail || '';
  processCompleteBadge.hidden = !success;
  processPercent.hidden = false;
  processPercent.textContent = success ? '100%' : '0%';
  setProcessStep(success ? 'complete' : 'failed');
  setConsoleMode(success ? 'READY' : 'ERROR');
}

function showUploadMessage(html, isError = false) {
  uploadResult.hidden = false;
  uploadResult.classList.toggle('error', isError);
  uploadResult.innerHTML = html;
}

function ruleList(rules, emptyLabel) {
  const items = Array.isArray(rules) ? rules.filter(Boolean) : [];
  if (!items.length) return `<span class="muted">${emptyLabel}</span>`;
  return `<ul>${items.map((rule) => `<li>${escapeHtml(rule)}</li>`).join('')}</ul>`;
}

function renderUploadResult(data) {
  const videoId = escapeHtml(data.video_id || 'unknown');
  const frameCount = Number(data.frames_extracted || 0).toLocaleString();
  const source = escapeHtml(data.rules_source || 'none');

  showUploadMessage(`
    <div class="result-metrics">
      <span><strong>${videoId}</strong><small>Video ID</small></span>
      <span><strong>${frameCount}</strong><small>Frames</small></span>
      <span><strong>${source}</strong><small>Rules</small></span>
    </div>
    <div class="result-rules">
      <div><b>Normal</b>${ruleList(data.generated_normal_rules, 'No generated normal rules')}</div>
      <div><b>Abnormal</b>${ruleList(data.generated_abnormal_rules, 'No generated abnormal rules')}</div>
    </div>
  `);
}

async function readErrorMessage(response) {
  let message = await response.text();
  try {
    const parsed = JSON.parse(message);
    message = parsed.detail || message;
  } catch (_) {
    // Keep the raw response body.
  }
  return message || `Request failed: ${response.status}`;
}

async function startPrepareJob(formData) {
  const response = await fetch('/api/prepare/start', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function fetchPrepareProgress(jobId) {
  const response = await fetch(`/api/prepare/${encodeURIComponent(jobId)}/progress`, {
    cache: 'no-store',
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function waitForPrepareJob(jobId) {
  while (true) {
    const job = await fetchPrepareProgress(jobId);
    renderPrepareProgress(job);

    if (job.status === 'complete') {
      return job.result;
    }

    if (job.status === 'failed') {
      throw new Error(job.error || job.message || 'Preparation failed');
    }

    await sleep(5000);
  }
}

async function prepareVideo(event) {
  event.preventDefault();
  const file = videoFile.files && videoFile.files[0];

  if (!file) {
    showUploadMessage('<span class="muted">Select a video file before preparing.</span>', true);
    return;
  }

  const formData = new FormData();
  formData.append('video_file', file);
  formData.append('video_id', videoIdInput.value.trim());
  formData.append('sample_every_n', sampleEvery.value || '8');
  formData.append('generate_video_rules', generateRules.checked ? 'true' : 'false');
  formData.append('normal_rules', normalRules.value.trim());
  formData.append('abnormal_rules', abnormalRules.value.trim());
  formData.append('rule_context_query', ruleContext.value.trim());

  prepareBtn.disabled = true;
  videoFile.disabled = true;
  uploadResult.hidden = true;
  startProcess('Uploading video', 'Sending file to server');

  try {
    const { job_id: jobId } = await startPrepareJob(formData);
    const data = await waitForPrepareJob(jobId);
    updateActiveVideo(data.video_id, file.name);
    await loadVideoRules(data.video_id);
    renderUploadResult(data);
    renderEvidence([]);
    answerText.innerHTML = `Prepared <span class="hl">${escapeHtml(data.video_id)}</span>. Ask a question to retrieve evidence from this video.`;
    answerText.dataset.baseAnswer = answerText.innerHTML;
    confVal.textContent = 'READY';
    confFill.style.width = '100%';
    stopProcess(true, 'Video indexed', data);
  } catch (error) {
    stopProcess(false, 'Preparation failed');
    showUploadMessage(`<span class="muted">${escapeHtml(error.message)}</span>`, true);
    answerText.innerHTML = 'Preparation failed. Check the upload settings and backend logs.';
    confVal.textContent = 'ERROR';
    confFill.style.width = '10%';
    console.error(error);
  } finally {
    prepareBtn.disabled = false;
    videoFile.disabled = false;
  }
}

async function askBackend(question) {
  const response = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      video_id: activeVideoId || undefined,
    }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function submitQuestion() {
  const question = askInput.value.trim();
  if (!question) return;

  askSend.disabled = true;
  askSend.classList.add('is-loading');
  setConsoleMode('ANSWERING');
  answerText.innerHTML = `Analyzing <span class="hl">${escapeHtml(activeVideoId || 'indexed videos')}</span> for: <span class="hl">${escapeHtml(question)}</span><span class="typing-dots"><i></i><i></i><i></i></span>`;
  answerText.dataset.baseAnswer = '';
  confVal.textContent = '...';
  confFill.style.width = '35%';

  try {
    const data = await askBackend(question);
    const classification = data.classification ? ` <span class="hl">[${escapeHtml(data.classification)}]</span>` : '';
    answerText.innerHTML = `${escapeHtml(data.answer || 'No answer returned.')}${classification}`;
    answerText.dataset.baseAnswer = answerText.innerHTML;

    const confidence = typeof data.confidence === 'number' ? data.confidence : null;
    if (confidence !== null) {
      confVal.textContent = `${Math.round(confidence * 100)}%`;
      confFill.style.width = `${clamp(confidence * 100, 3, 100)}%`;
    } else {
      confVal.textContent = 'N/A';
      confFill.style.width = '10%';
    }

    renderEvidence(data.evidence || []);
    setConsoleMode('READY');
  } catch (error) {
    answerText.innerHTML = `Could not reach <span class="hl">POST /ask</span>. ${escapeHtml(error.message)}`;
    answerText.dataset.baseAnswer = answerText.innerHTML;
    confVal.textContent = 'ERROR';
    confFill.style.width = '10%';
    setConsoleMode('ERROR');
    console.error(error);
  } finally {
    askSend.disabled = false;
    askSend.classList.remove('is-loading');
  }
}

if (uploadForm) uploadForm.addEventListener('submit', prepareVideo);
if (grid) grid.addEventListener('click', handleEvidenceGridClick);
if (editRulesButton) editRulesButton.addEventListener('click', () => openRulesEditor('normal'));
if (addNormalRuleButton) addNormalRuleButton.addEventListener('click', () => openRulesEditor('normal'));
if (addAbnormalRuleButton) addAbnormalRuleButton.addEventListener('click', () => openRulesEditor('abnormal'));
if (cancelRuleEditButton) cancelRuleEditButton.addEventListener('click', closeRulesEditor);
if (rulesEditor) rulesEditor.addEventListener('submit', saveRuleEdits);
if (rulesPanel) rulesPanel.addEventListener('click', handleRulePanelClick);

renderAppliedRules(null);

if (videoFile) {
  videoFile.addEventListener('change', () => {
    const file = videoFile.files && videoFile.files[0];
    fileName.textContent = file ? file.name : 'Video file';
  });
}

if (dropzone) {
  ['dragenter', 'dragover'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add('dragging');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove('dragging');
    });
  });

  dropzone.addEventListener('drop', (event) => {
    if (!event.dataTransfer.files.length) return;
    videoFile.files = event.dataTransfer.files;
    fileName.textContent = event.dataTransfer.files[0].name;
  });
}

document.querySelectorAll('[data-focus-upload]').forEach((link) => {
  link.addEventListener('click', (event) => {
    event.preventDefault();
    document.getElementById('uploadPanel')?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    });
    history.replaceState(null, '', '#demo');
    setTimeout(() => videoFile?.focus({ preventScroll: true }), 450);
  });
});

document.querySelectorAll('.ask-chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    askInput.value = chip.textContent;
    askInput.focus();
  });
});

askSend.addEventListener('click', submitQuestion);
askInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') submitQuestion();
});

// ---- scroll reveal ----
const io = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      const siblings = [...entry.target.parentElement.children].filter((child) => child.classList.contains('reveal'));
      const position = siblings.indexOf(entry.target);
      entry.target.style.transitionDelay = `${position * 0.08}s`;
      entry.target.classList.add('in');
      io.unobserve(entry.target);
    }
  });
}, { threshold: 0.15 });

document.querySelectorAll('.reveal').forEach((element) => io.observe(element));
