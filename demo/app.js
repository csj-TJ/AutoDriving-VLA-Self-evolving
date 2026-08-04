const $ = selector => document.querySelector(selector);
const form = $('#form');
const canvas = $('#scene');
const ctx = canvas.getContext('2d');
const numericFields = ['ego_speed', 'speed_limit', 'lead_distance', 'lead_speed', 'stopline_distance', 'pedestrian_distance', 'road_curvature'];
const numericFieldLabels = {
  ego_speed: '自车速度',
  speed_limit: '道路限速',
  lead_distance: '前车距离',
  lead_speed: '前车速度',
  stopline_distance: '停止线距离',
  pedestrian_distance: '行人距离',
  road_curvature: '道路曲率',
};
const sceneFields = [...numericFields, 'weather', 'route_command', 'traffic_light'];

const state = {
  data: null,
  frame: 0,
  playing: false,
  lastFrameAt: 0,
  sceneId: '',
  sampleId: '',
  scenarioSource: 'training_dataset',
  dirty: false,
  dataTotal: 0,
  cameraImageUrl: '',
};

function escapeHtml(value) {
  const node = document.createElement('div');
  node.textContent = String(value ?? '');
  return node.innerHTML;
}

function showError(message) {
  const box = $('#error');
  box.textContent = message;
  box.style.display = 'block';
  setTimeout(() => { box.style.display = 'none'; }, 5000);
}

async function api(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok || body.error) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function setCameraPreview(url = '') {
  state.cameraImageUrl = url || '';
  const preview = $('#cameraPreview');
  const image = $('#cameraImage');
  preview.classList.toggle('visible', Boolean(url));
  image.onerror = () => preview.classList.remove('visible');
  image.onload = () => preview.classList.add('visible');
  if (url) image.src = url;
  else image.removeAttribute('src');
}

function applyScenario(scenario, sampleId = '', source = 'training_dataset', runNow = true, cameraUrl = '') {
  state.sceneId = scenario.scene_id;
  state.sampleId = sampleId;
  state.scenarioSource = source;
  state.dirty = false;
  Object.entries(scenario).forEach(([key, value]) => {
    if (form.elements[key]) form.elements[key].value = value;
  });
  form.elements.source_sample_id.value = sampleId;
  $('#sceneIdInput').value = scenario.scene_id;
  $('#sceneName').textContent = `${scenario.scene_id}${sampleId ? ` · ${sampleId}` : ''}`;
  setCameraPreview(cameraUrl);
  $('#run').disabled = false;
  $('#run').textContent = '运行本轮分析';
  if (runNow) run();
}

function formPayload() {
  const payload = Object.fromEntries(new FormData(form));
  numericFields.forEach(key => { payload[key] = Number(payload[key]); });
  payload.scene_id = state.dirty ? `${state.sceneId}:edited` : state.sceneId;
  payload.scenario_source = state.dirty ? 'user_control_modified' : state.scenarioSource;
  payload.source_sample_id = state.sampleId || null;
  return payload;
}

function validateNumericFields() {
  for (const key of numericFields) {
    const input = form.elements[key];
    const raw = input.value.trim();
    if (raw === '' || !Number.isFinite(Number(raw))) {
      input.focus();
      showError(`${numericFieldLabels[key]}请输入有效数字`);
      return false;
    }
  }
  return true;
}

async function loadSceneById(sceneId) {
  if (!sceneId.trim()) return showError('请输入训练集 Scene ID');
  const result = await api(`/api/scenario?id=${encodeURIComponent(sceneId.trim())}`);
  applyScenario(result.scenario, result.sample_id, 'training_dataset', true, result.camera_image_url);
}

async function loadRandomScene() {
  if (!state.dataTotal) return showError('训练数据索引尚未加载');
  const offset = Math.floor(Math.random() * state.dataTotal);
  const page = await api(`/api/scenarios?offset=${offset}&limit=1`);
  if (!page.items.length) return showError('未找到训练样本');
  const row = page.items[0];
  applyScenario(row.scenario, row.sample_id, 'training_dataset_random', true, row.camera_image_url);
}

async function loadPresets() {
  const result = await api('/api/scenarios/presets');
  state.dataTotal = result.data.records;
  $('#dataStatus').innerHTML = `已索引 <b>${result.data.records.toLocaleString()}</b> 条完整记录 · ${(result.data.bytes / 1048576).toFixed(2)} MB · ${escapeHtml(Object.entries(result.data.splits).map(([k, v]) => `${k}:${v}`).join(' / '))}`;
  const box = $('#presets');
  box.innerHTML = result.items.map((item, index) => `<button type="button" data-index="${index}" class="${index === 0 ? 'active' : ''}">${escapeHtml(item.label)}</button>`).join('');
  box.querySelectorAll('button').forEach(button => {
    button.onclick = () => {
      box.querySelectorAll('button').forEach(item => item.classList.toggle('active', item === button));
      const preset = result.items[Number(button.dataset.index)];
      applyScenario(preset.scenario, preset.sample_id, `training_preset:${preset.key}`, true, preset.camera_image_url);
    };
  });
  if (!result.items.length) throw new Error('训练数据中没有可用预设场景');
  applyScenario(result.items[0].scenario, result.items[0].sample_id, `training_preset:${result.items[0].key}`, true, result.items[0].camera_image_url);
}

async function run() {
  if (!validateNumericFields()) return;
  const button = $('#run');
  button.disabled = true;
  button.textContent = '正在规划、诊断与重规划…';
  try {
    const result = await api('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formPayload()),
    });
    state.data = result;
    state.frame = 0;
    state.playing = true;
    renderPanels();
    draw();
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = '运行本轮分析';
  }
}

function list(items) {
  return items?.length
    ? `<ul>${items.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`
    : '<p class="empty">无硬约束失败</p>';
}

function renderPanels() {
  const data = state.data;
  const selected = data.selected_policy;
  const baseline = data.results.baseline;
  const revised = data.results[selected];
  const critic = revised.critic;
  const reflection = baseline.reflection;
  const baselineModel = baseline.model;
  const revisedModel = revised.model;
  const provenance = revised.score_provenance;
  const skill = data.generated_skill;

  $('#sceneName').textContent = `${data.scenario.scene_id} · 请求 ${data.request_id}`;
  $('#baseOverall').textContent = baseline.critic.overall_score.toFixed(1);
  $('#newOverall').textContent = critic.overall_score.toFixed(1);
  $('#baseTarget').textContent = `${baseline.trajectory.target_speed.toFixed(1)} m/s`;
  $('#newTarget').textContent = `${revised.trajectory.target_speed.toFixed(1)} m/s`;
  $('#baseFailures').textContent = `${baseline.critic.failures.length} 项`;
  $('#decision').textContent = data.delta.target_speed < -.2 ? '减速修正' : data.delta.target_speed > .2 ? '加速修正' : '保持策略';
  $('#delta').textContent = `${data.delta.overall >= 0 ? '+' : ''}${data.delta.overall.toFixed(1)} 综合分`;

  const scores = [['Safety', critic.safety_score], ['Rule', critic.rule_score], ['Comfort', critic.comfort_score], ['Overall', critic.overall_score]];
  $('#scores').innerHTML = scores.map(([name, value]) => `<div class="score"><span>${name}</span><strong>${value.toFixed(1)}</strong><div class="bar"><i style="width:${value}%"></i></div></div>`).join('');
  $('#scoreSource').innerHTML = `<b>${escapeHtml(critic.critic_type)}</b><br>规则实时轨迹评分 65% · Reward Critic 25% · 全训练集近邻 10%<br>近邻：${provenance.neighbors.map(item => escapeHtml(item.sample_id)).join('、')}<br>评分耗时：${provenance.latency_ms.toFixed(2)} ms`;

  renderSkill(skill);

  $('#timeline').innerHTML = data.events.map((event, index) => `<div class="event ${index === 0 ? 'active' : ''}"><b>${String(index + 1).padStart(2, '0')} · ${escapeHtml(event.phase)}</b><p>${escapeHtml(event.detail)}</p></div>`).join('');
  const history = data.evolution_history || [];
  $('#evolution').innerHTML = history.length
    ? history.map((entry, index) => `<div class="event ${index === history.length - 1 ? 'active' : ''}"><b>Round ${entry.round} · 综合分 ${entry.mean_overall.toFixed(1)}</b><p>反思 ${entry.revised_samples} 条 · 失败事件 ${entry.failure_events} · 记忆池 ${entry.memory_size}</p></div>`).join('')
    : '<div class="empty">尚未运行本地自进化训练。</div>';
  $('#reflection').innerHTML = `<span class="verdict">${reflection.verdict === 'revise' ? '需要修正' : '轨迹可接受'}</span><h3>根因</h3>${list(reflection.root_causes)}<h3>证据</h3>${list(reflection.evidence)}<h3>纠正策略</h3>${list(reflection.corrective_strategy)}<h3>反事实动作</h3><ul><li>${escapeHtml(reflection.counterfactual_action)}</li></ul>`;

  const usingCache = [baselineModel, revisedModel].some(model => model.runtime_mode === 'opendrivevla_cache');
  $('#runtimePill').innerHTML = `<i></i>${usingCache ? 'OpenDriveVLA 真实缓存' : 'LiteVLA 实时回退'}`;
  $('#model').innerHTML = `<strong>OpenDriveVLA-0.5B · ${revisedModel.checkpoint_installed ? 'checkpoint 已核验' : 'checkpoint 不完整'}</strong>请求模式：${escapeHtml(form.elements.runtime.value)}<br>Baseline 实际：${escapeHtml(baselineModel.runtime)} · ${data.provenance.timing.baseline.model_latency_ms.toFixed(2)} ms<br>${escapeHtml(selected)} 实际：${escapeHtml(revisedModel.runtime)} · ${data.provenance.timing[selected].model_latency_ms.toFixed(2)} ms<br>数据：${data.provenance.data.records.toLocaleString()} 条完整索引 · 来源 ${escapeHtml(data.provenance.scenario_source)}<br>${escapeHtml(revisedModel.disclosure)}`;
}

function renderSkill(skill) {
  if (!skill) return;
  const statusText = skill.status === 'validated' ? '已通过同场景复评' : '候选 Skill';
  $('#skillStatus').textContent = `${statusText} · 置信度 ${(skill.confidence * 100).toFixed(0)}%`;
  $('#skillName').textContent = skill.name;
  $('#skillId').textContent = skill.skill_id;
  $('#skillCamera').src = state.cameraImageUrl || '';
  $('#skillCamera').classList.toggle('visible', Boolean(state.cameraImageUrl));
  $('#skillCamera').onerror = () => $('#skillCamera').classList.remove('visible');
  const evidence = skill.evidence_source;
  const token = evidence.annotation_token || evidence.sample_token || '交互参数场景';
  $('#skillSource').textContent = `${evidence.dataset} · ${evidence.camera_count || 0} 路相机 · 证据 ${token}`;
  $('#skillFlow').innerHTML = skill.generation_stages.map((stage, index) => `
    <div class="skill-step" data-step="${index}">
      <span>${String(index + 1).padStart(2, '0')}</span>
      <b>${escapeHtml(stage.phase)}</b>
      <em>${escapeHtml(stage.artifact)}</em>
      <p>${escapeHtml(stage.detail)}</p>
    </div>`).join('');
  $('#skillTriggers').innerHTML = skill.triggers.map(item => `
    <div><span>${escapeHtml(item.label)}</span><b>${escapeHtml(item.value)}</b></div>`).join('');
  $('#skillActions').innerHTML = skill.actions.map(action => `<li>${escapeHtml(action)}</li>`).join('');
  const validation = skill.validation;
  const rounds = skill.memory.round_support.map(item => `R${item.round}:${item.records}`).join(' · ') || '本轮首次生成';
  $('#skillValidation').innerHTML = `
    <strong>${validation.passed ? '✓ Skill 验证通过' : '△ 保留为候选'}</strong>
    <span>综合分 ${validation.overall_delta >= 0 ? '+' : ''}${validation.overall_delta.toFixed(1)}</span>
    <span>安全分 ${validation.safety_delta >= 0 ? '+' : ''}${validation.safety_delta.toFixed(1)}</span>
    <span>目标速度 ${validation.target_speed_delta >= 0 ? '+' : ''}${validation.target_speed_delta.toFixed(1)} m/s</span>
    <span>记忆支持 ${skill.memory.matched_records} 条 · ${escapeHtml(rounds)}</span>`;
  updateSkillFlow(0);
}

function updateSkillFlow(activeIndex) {
  document.querySelectorAll('#skillFlow .skill-step').forEach((step, index) => {
    step.classList.toggle('complete', index < activeIndex);
    step.classList.toggle('active', index === activeIndex);
  });
}

function worldPoint(point) {
  const horizon = state.data?.visualization.world_horizon_m || 70;
  const longitudinalScale = (canvas.height - 72) / horizon;
  const lateralScale = 25;
  return { x: canvas.width / 2 + point[1] * lateralScale, y: canvas.height - 38 - point[0] * longitudinalScale };
}

function rounded(x, y, width, height, radius, fill) {
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, radius);
  ctx.fill();
}

function polyline(points, offset, stroke, width, dashed = false) {
  ctx.beginPath();
  points.forEach((point, index) => {
    const p = worldPoint([point[0], point[1] + offset]);
    index ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
  });
  ctx.setLineDash(dashed ? [18, 15] : []);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawRoad() {
  const scenario = state.data.scenario;
  const visual = state.data.visualization;
  const night = scenario.weather === 'night';
  ctx.fillStyle = night ? '#040b12' : '#0d1a24';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const half = visual.road_width_m / 2;
  const left = visual.centerline.map(point => worldPoint([point[0], point[1] - half]));
  const right = [...visual.centerline].reverse().map(point => worldPoint([point[0], point[1] + half]));
  ctx.beginPath();
  [...left, ...right].forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
  ctx.closePath();
  ctx.fillStyle = night ? '#202a33' : '#303b45';
  ctx.fill();
  polyline(visual.centerline, -half, '#101820', 5);
  polyline(visual.centerline, half, '#101820', 5);
  [-half / 3, half / 3].forEach(offset => polyline(visual.centerline, offset, '#d8e0e777', 2, true));

  const stop = visual.stop_line;
  const stopLeft = worldPoint([stop.x, stop.center_y - half]);
  const stopRight = worldPoint([stop.x, stop.center_y + half]);
  ctx.strokeStyle = '#f4f7fa';
  ctx.lineWidth = 7;
  ctx.beginPath(); ctx.moveTo(stopLeft.x, stopLeft.y); ctx.lineTo(stopRight.x, stopRight.y); ctx.stroke();

  const pedestrianTrack = Array.isArray(visual.pedestrian?.track)
    ? visual.pedestrian.track
    : [];

  if (visual.pedestrian?.visible && pedestrianTrack.length > 1) {
    ctx.strokeStyle = '#ffcf6d88'; ctx.lineWidth = 2; ctx.setLineDash([5, 5]);
    ctx.beginPath();
    pedestrianTrack.forEach((point, index) => {
      const pixel = worldPoint([point.x, point.y]);
      if (index === 0) ctx.moveTo(pixel.x, pixel.y); else ctx.lineTo(pixel.x, pixel.y);
    });
    ctx.stroke(); ctx.setLineDash([]);
  }

  const signal = worldPoint([visual.traffic_light.x, visual.traffic_light.y]);
  rounded(signal.x - 11, signal.y - 31, 22, 62, 7, '#14212d');
  [['red', -16, '#ff5268'], ['green', 16, '#4ee398']].forEach(([name, dy, color]) => {
    ctx.fillStyle = scenario.traffic_light === name ? color : '#294039';
    ctx.beginPath(); ctx.arc(signal.x, signal.y + dy, 6, 0, Math.PI * 2); ctx.fill();
  });

  const animationTime = Math.min(state.frame, 11) * .5;
  if (visual.lead_vehicle.visible) {
    const leadX = visual.lead_vehicle.x + visual.lead_vehicle.speed * animationTime;
    const leadY = scenario.road_curvature * (Math.max(0, leadX) ** 1.45) * .12;
    const lead = worldPoint([leadX, leadY]);
    const leadAhead = worldPoint([leadX + 1, scenario.road_curvature * ((leadX + 1) ** 1.45) * .12]);
    const leadAngle = Math.atan2(leadAhead.y - lead.y, leadAhead.x - lead.x) + Math.PI / 2;
    drawCar(lead.x, lead.y, '#d98b37', leadAngle);
    ctx.fillStyle = '#ffd69a'; ctx.font = '12px sans-serif';
    ctx.fillText(`前车 ${visual.lead_vehicle.speed.toFixed(1)}m/s`, lead.x + 18, lead.y);
  }
  if (visual.pedestrian?.visible && pedestrianTrack.length > 0) {
    const pedestrianPoint = interpolateTrack(pedestrianTrack, animationTime);
    drawPedestrian(worldPoint([pedestrianPoint.x, pedestrianPoint.y]),
      animationTime * Math.max(.2, visual.pedestrian.mean_speed_mps || 0));
    ctx.fillStyle = '#ffe29b'; ctx.font = '12px sans-serif';
    const sourceLabel = visual.pedestrian.data_source === 'nuscenes_sample_annotation' ? 'nuScenes 标注' : '静态场景值';
    const pedestrianPixel = worldPoint([pedestrianPoint.x, pedestrianPoint.y]);
    ctx.fillText(`行人 ${visual.pedestrian.mean_speed_mps.toFixed(1)}m/s · ${sourceLabel}`,
      pedestrianPixel.x + 12, pedestrianPixel.y - 10);
  }

  ctx.fillStyle = '#dcebf7'; ctx.font = '600 13px sans-serif';
  const command = { straight: '↑ 直行', left: '↖ 左转', right: '↗ 右转' }[visual.route_command];
  ctx.fillText(`${command} · ${scenario.weather} · 曲率 ${scenario.road_curvature.toFixed(3)}`, 20, 28);

  if (scenario.weather === 'rain') drawRain();
  if (scenario.weather === 'fog') {
    const fog = ctx.createLinearGradient(0, 0, 0, canvas.height);
    fog.addColorStop(0, '#dce8ee77'); fog.addColorStop(1, '#b7c5cc12');
    ctx.fillStyle = fog; ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
}

function interpolateTrack(points, time) {
  if (!points.length) return { x: 0, y: 0 };
  if (time <= points[0].t) return points[0];
  for (let index = 1; index < points.length; index += 1) {
    if (time <= points[index].t) {
      const previous = points[index - 1];
      const next = points[index];
      const ratio = (time - previous.t) / Math.max(.001, next.t - previous.t);
      return {
        x: previous.x + (next.x - previous.x) * ratio,
        y: previous.y + (next.y - previous.y) * ratio,
      };
    }
  }
  return points[points.length - 1];
}

function drawRain() {
  ctx.strokeStyle = '#76b5e477'; ctx.lineWidth = 1;
  for (let index = 0; index < 70; index++) {
    const x = (index * 71 + state.frame * 17) % canvas.width;
    const y = (index * 43 + state.frame * 23) % canvas.height;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - 8, y + 18); ctx.stroke();
  }
}

function drawPedestrian(point, time = 0) {
  const stride = Math.sin(time * 7) * 5;
  ctx.fillStyle = '#ffd56a';
  ctx.beginPath(); ctx.arc(point.x, point.y - 8, 5, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#ffd56a'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(point.x, point.y - 2); ctx.lineTo(point.x, point.y + 12);
  ctx.moveTo(point.x, point.y + 4); ctx.lineTo(point.x - 7, point.y + 10 + stride);
  ctx.moveTo(point.x, point.y + 4); ctx.lineTo(point.x + 7, point.y + 10 - stride);
  ctx.moveTo(point.x, point.y + 12); ctx.lineTo(point.x - 5, point.y + 20 - stride);
  ctx.moveTo(point.x, point.y + 12); ctx.lineTo(point.x + 5, point.y + 20 + stride); ctx.stroke();
}

function drawCar(x, y, color, angle = 0) {
  ctx.save(); ctx.translate(x, y); ctx.rotate(angle);
  rounded(-14, -25, 28, 50, 7, color);
  ctx.fillStyle = '#9ed5ec'; ctx.fillRect(-9, -14, 18, 10);
  ctx.fillStyle = '#111a22'; ctx.fillRect(-9, 10, 18, 8);
  ctx.restore();
}

function drawTrajectory(points, color, width) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath();
  points.forEach((point, index) => {
    const p = worldPoint(point);
    index ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
  });
  ctx.stroke();
}

function trajectoryHeading(points, index) {
  const current = worldPoint(points[index]);
  for (let distance = 1; distance < points.length; distance += 1) {
    const forwardIndex = index + distance;
    if (forwardIndex < points.length) {
      const forward = worldPoint(points[forwardIndex]);
      if (Math.hypot(forward.x - current.x, forward.y - current.y) > .5) {
        return Math.atan2(forward.y - current.y, forward.x - current.x) + Math.PI / 2;
      }
    }
    const backwardIndex = index - distance;
    if (backwardIndex >= 0) {
      const backward = worldPoint(points[backwardIndex]);
      if (Math.hypot(current.x - backward.x, current.y - backward.y) > .5) {
        return Math.atan2(current.y - backward.y, current.x - backward.x) + Math.PI / 2;
      }
    }
  }
  return 0;
}

function draw() {
  if (!state.data) return;
  const selected = state.data.selected_policy;
  const baseline = state.data.results.baseline.trajectory.points;
  const revised = state.data.results[selected].trajectory.points;
  drawRoad();
  drawTrajectory(baseline, '#ff6376', 3);
  drawTrajectory(revised, '#35d8d0', 4);
  const index = Math.min(revised.length - 1, Math.floor(state.frame));
  const current = worldPoint(revised[index]);
  drawCar(current.x, current.y, '#35d8d0', trajectoryHeading(revised, index));
  $('#hudSpeed').innerHTML = `${revised[index][2].toFixed(1)} <small>m/s</small>`;
  $('#progress').style.width = `${state.frame / (revised.length - 1) * 100}%`;
  const eventCount = state.data.events.length;
  const active = Math.min(eventCount - 1, Math.floor(state.frame / Math.max(1, revised.length - 1) * eventCount));
  document.querySelectorAll('#timeline .event').forEach((event, i) => event.classList.toggle('active', i <= active));
  const skillStages = state.data.generated_skill?.generation_stages?.length || 1;
  updateSkillFlow(Math.min(skillStages - 1, Math.floor(state.frame / Math.max(1, revised.length - 1) * skillStages)));
}

function animationTick(timestamp) {
  if (state.data && state.playing && timestamp - state.lastFrameAt > 150) {
    const count = state.data.results[state.data.selected_policy].trajectory.points.length;
    state.frame += .32;
    if (state.frame >= count - 1) {
      state.frame = count - 1;
      state.playing = false;
      $('#play').textContent = '▶';
    }
    draw();
    state.lastFrameAt = timestamp;
  }
  requestAnimationFrame(animationTick);
}

sceneFields.forEach(name => form.elements[name].addEventListener('input', () => {
  if (!state.sceneId) return;
  state.dirty = true;
  state.scenarioSource = 'user_control_modified';
  $('#sceneName').textContent = `${state.sceneId} · 已修改控制参数`;
}));
form.onsubmit = event => { event.preventDefault(); run(); };
$('#loadScene').onclick = () => loadSceneById($('#sceneIdInput').value).catch(error => showError(error.message));
$('#randomScene').onclick = () => loadRandomScene().catch(error => showError(error.message));
$('#play').onclick = () => {
  if (!state.data) return;
  state.playing = !state.playing;
  const count = state.data.results[state.data.selected_policy].trajectory.points.length;
  if (state.frame >= count - 1 && state.playing) state.frame = 0;
  $('#play').textContent = state.playing ? 'Ⅱ' : '▶';
};
$('#reset').onclick = () => { state.frame = 0; state.playing = false; $('#play').textContent = '▶'; draw(); };

async function boot() {
  try {
    const meta = await api('/api/meta');
    const audit = meta.audit;
    $('#checkpointPill').innerHTML = `<i></i>${audit.checkpoint_installed && audit.code_complete ? '基座文件完整' : '基座文件待补'}`;
    await loadPresets();
  } catch (error) {
    showError(`初始化失败：${error.message}`);
    $('#dataStatus').textContent = error.message;
  }
  requestAnimationFrame(animationTick);
}

boot();
