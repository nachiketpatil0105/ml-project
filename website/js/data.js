'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const number = value => new Intl.NumberFormat('en-US').format(value);
  const percent = value => new Intl.NumberFormat('en-US', {style: 'percent', maximumFractionDigits: 2}).format(value);
  let analysis;
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function stat(value, label) {
    const node = element('div');
    node.append(element('strong', number(value)), element('span', label));
    return node;
  }
  function renderTarget() {
    const key = $('target-select').value;
    const target = analysis.target_summary[key];
    const meanings = {
      skip_2: 'The project target: the supplied brief-play skip indicator. Exact time cutoffs are not encoded in these CSV files.',
      skip_1: 'An alternative very-brief-play indicator supplied in the dataset. It is a different target from skip_2.',
      skip_3: 'An alternative supplied skip threshold. Do not infer an exact duration or interpret it as the same event as skip_2.',
      not_skipped: 'The complete-playback indicator. A true value means the track was played in its entirety; this is not the complement of skip_2.'
    };
    $('label-name').textContent = key;
    $('label-rate').textContent = percent(target.positive_rate);
    $('label-count').textContent = `${number(target.positive_count)} of ${number(target.labelled_plays)} labelled plays`;
    $('label-meaning').textContent = meanings[key];
    $('label-meter').value = target.positive_rate;
    $('label-meter').setAttribute('aria-label', `${key} is true for ${percent(target.positive_rate)} of labelled plays`);
  }
  function renderChart() {
    const index = Number($('chart-select').value);
    const chart = analysis.charts[index];
    $('chart-position').textContent = `${index + 1} / ${analysis.charts.length} views`;
    $('chart-title').textContent = chart.title;
    $('chart-caption').textContent = chart.caption;
    $('chart-image').alt = chart.alt;
    $('chart-image').src = chart.image_url;
    $('chart-open').href = chart.image_url;
    $('chart-download').href = chart.image_url;
    $('chart-insight').textContent = chart.insight;
  }
  function renderVariables() {
    const query = $('variable-search').value.toLowerCase().trim();
    const rows = analysis.important_variables.filter(v => `${v.name} ${v.table} ${v.meaning} ${v.relevance}`.toLowerCase().includes(query));
    $('variable-rows').replaceChildren(...rows.map(v => {
      const row = element('tr');
      const heading = element('th');
      heading.scope = 'row';
      heading.append(element('code', v.name), element('small', v.table === 'log' ? 'Session log' : 'Track features'));
      row.append(heading, element('td', v.meaning), element('td', v.relevance));
      return row;
    }));
    $('variable-count').textContent = `${rows.length} of ${analysis.important_variables.length} variable groups${rows.length ? '' : ' — no matches; try another term'}. Full column summaries are available in the JSON.`;
  }
  function render(data) {
    analysis = data;
    const size = data.size, quality = data.data_quality, checks = quality.checks;
    $('overview-stats').replaceChildren(stat(size.track_plays, 'track plays'), stat(size.sessions, 'listening sessions'), stat(size.metadata_tracks, 'unique metadata tracks'), stat(size.log_columns + size.track_columns, 'columns across two tables'));
    $('coverage').textContent = `${size.session_length_min}–${size.session_length_max} tracks per session · ${size.distinct_dates} distinct dates, ${size.date_min} to ${size.date_max}. ${size.note}`;
    $('input-tables').replaceChildren(...data.tables.map(t => {
      const panel = element('article', undefined, 'panel');
      panel.append(element('p', t.file, 'eyebrow'), element('h3', t.row_unit), element('p', `${number(t.rows)} rows · ${t.columns} columns · ${(t.size_bytes / 1e6).toFixed(1)} MB`));
      return panel;
    }));
    $('join-note').textContent = `Join ${data.join.left} → ${data.join.right} (${data.join.cardinality}). Unmatched track-play rows: ${number(data.join.unmatched_rows)}.`;
    $('source-links').replaceChildren(...data.sources.map(s => {
      const card = element('div');
      const link = element('a', `${s.title} ↗`);
      link.href = s.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
      card.append(link, element('p', s.role));
      return card;
    }));
    $('access-limits').replaceChildren(...data.access_limits.map(note => element('li', note)));
    $('definition-note').textContent = `${number(quality.observations.non_skip2_not_complete_play_rows)} plays have both skip_2=false and not_skipped=false. The primary plots use skip_2; switching the label above changes only the label summary, not the plots.`;
    $('chart-select').replaceChildren(...data.charts.map((chart, i) => {
      const option = element('option', chart.title); option.value = i; return option;
    }));
    $('key-findings').replaceChildren(...data.key_findings.map(f => {
      const card = element('article', undefined, 'finding');
      card.append(element('h3', f.title), element('p', f.text)); return card;
    }));
    $('quality-status').textContent = quality.status === 'no_structural_problems_detected'
      ? 'No structural problems were detected by the implemented checks. That does not make every value or assumption problem-free: the cases below still matter for modelling.'
      : 'The analysis detected issues requiring review. Inspect the checks and source data before modelling.';
    const missing = Object.values(checks.missing_by_column).reduce((total, columns) => total + Object.values(columns).reduce((a, b) => a + b, 0), 0);
    $('quality-stats').replaceChildren(stat(missing, 'missing cells'), stat(checks.duplicate_session_positions, 'duplicate session positions'), stat(checks.unmatched_log_tracks_rows, 'unmatched track-play rows'), stat(checks.invalid_sessions, 'invalid sessions'));
    $('quality-issues').replaceChildren(...quality.issues.map(issue => {
      const card = element('article', undefined, 'issue-card');
      card.append(element('span', issue.severity.replaceAll('_', ' '), 'issue-type'), element('h3', issue.title), element('strong', `${number(issue.count)} ${issue.unit}`, 'issue-count'), element('p', issue.detail));
      return card;
    }));
    $('quality-action').textContent = quality.action;
    $('scope-note').textContent = `${data.scope.population}. ${data.scope.purpose}. ${data.scope.split_status}.`;
    $('scope-caveat').textContent = data.scope.caveat;
    $('run-command').textContent = data.reproduce.command;
    $('generated-note').textContent = `Generated: ${new Date(data.generated_at).toLocaleString('en-GB', {dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC'})} UTC. Figure and summary values are loaded from the same JSON.`;
    renderTarget(); renderChart(); renderVariables();
    $('analysis-content').hidden = false;
    $('load-status').textContent = 'Analysis loaded from the generated results file.';
    $('load-status').className = 'source-note';
  }
  async function load() {
    $('retry-load').hidden = true;
    $('load-status').textContent = 'Loading the analysis…';
    try {
      const response = await fetch('../../python/01_data_understanding/results/data_understanding.json');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (data.schema_version !== '1.0' || !data.charts?.length || !data.size || !data.data_quality) throw new Error('Unsupported analysis format');
      render(data);
    } catch (error) {
      $('analysis-content').hidden = true;
      $('load-status').textContent = 'The analysis could not be loaded. Serve the project over HTTP (rather than opening file://), and check that python/01_data_understanding/results/data_understanding.json is available.';
      $('retry-load').hidden = false;
      console.error('Data page:', error);
    }
  }
  $('target-select').addEventListener('change', renderTarget);
  $('chart-select').addEventListener('change', renderChart);
  $('variable-search').addEventListener('input', renderVariables);
  $('retry-load').addEventListener('click', load);
  load();
})();
