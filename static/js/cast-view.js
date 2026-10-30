/* cast-view.js — Generate screen Cast list and Edit-speaker drawer.
   Owned by the cast screen agent of the "Estudio" redesign. Classic script, loaded last; no top-level DOM access.

   Renders #speakers-list (List / Details views) from main.js state: the detected
   speakers, the hidden per-speaker rows in #inline-voice-assignment-list (assigned
   voice), speakerReadyState, voiceFxState and currentStats.speaker_chunks.
   Every row is a `.speaker-tag[data-speaker]`, so main.js's delegated click opens
   the drawer (#speaker-edit-modal) and other screens can open a speaker the same way.

   main.js calls (guarded with typeof): renderCastList(speakers, stats) from
   displayStatistics, castViewRender() whenever a voice / ready / FX value changes,
   castDrawerOpened(speaker, row) and castDrawerClosed(speaker) around the drawer.
   Dispatches `cast:changed` on document: {speakers: [{name, color, ready, voiceLabel}]}. */

let castViewMode = 'list';
let castSpeakerNames = [];
let castStatsSnapshot = null;
let castRenderQueued = false;
let castLastSignature = '';
let castPreview = null;
let castDrawerReturnFocus = null;

/* ---------- state readers (main.js globals, read at call time) ---------- */

function castCurrentStats() {
    return typeof currentStats !== 'undefined' && currentStats ? currentStats : castStatsSnapshot;
}

function castEngineName() {
    const selected = typeof getSelectedJobEngine === 'function' ? getSelectedJobEngine() : '';
    const settings = typeof runtimeSettings !== 'undefined' ? runtimeSettings : null;
    return selected || settings?.tts_engine || 'kokoro';
}

function castEngineLabel() {
    const select = document.getElementById('job-tts-engine');
    const option = select?.selectedOptions?.[0];
    return option && option.value ? option.textContent.trim() : '';
}

function castSpeakerColor(name) {
    const ordered = castCurrentStats()?.speakers || castSpeakerNames;
    return typeof window.studioSpeakerColor === 'function'
        ? window.studioSpeakerColor(name, ordered)
        : '#c4b8a5';
}

// The row open in the drawer is the one being edited. A re-analysis while the
// drawer is open rebuilds a fresh (unassigned) copy in the hidden list, and
// closeSpeakerEditModal puts the edited row back in its place, so prefer it.
function castAssignmentRow(speaker) {
    const drawerRow = Array.from(document.querySelectorAll('#speaker-edit-modal-body .voice-assignment-row'))
        .find(row => row.dataset.speaker === speaker);
    if (drawerRow) return drawerRow;
    if (typeof getAssignmentRows !== 'function') return null;
    return getAssignmentRows().find(row => row.dataset.speaker === speaker) || null;
}

function castChunkCount(speaker) {
    const stats = castCurrentStats();
    if (!stats) return null;
    if (!stats.has_speaker_tags) {
        return Number.isFinite(Number(stats.total_chunks)) ? Number(stats.total_chunks) : null;
    }
    const counts = stats.speaker_chunks;
    if (!counts || typeof counts !== 'object') return null;
    const key = String(speaker || '').trim().toLowerCase();
    const value = counts[speaker] ?? counts[key];
    return Number.isFinite(Number(value)) ? Number(value) : 0;
}

function castFxValues(speaker) {
    const state = typeof voiceFxState !== 'undefined' ? voiceFxState[speaker] : null;
    return {
        pitch: Number(state?.pitch) || 0,
        speed: Number(state?.speed) || 1
    };
}

function castFormatDuration(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = String(total % 60).padStart(2, '0');
    return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${secs}` : `${minutes}:${secs}`;
}

/* ---------- voice summary ("voice · voiceName") ---------- */

function castSampleName(reference) {
    const value = String(reference || '').trim();
    if (!value) return '';
    const entry = typeof findChatterboxVoiceByPath === 'function' ? findChatterboxVoiceByPath(value) : null;
    if (entry?.name) return entry.name;
    return value.split(/[\\/]/).pop().replace(/\.[a-z0-9]{2,4}$/i, '');
}

function castCatalogDisplayName(select, value) {
    const option = Array.from(select?.options || []).find(item => item.value === value);
    let name = (option?.textContent || '').trim();
    if (name) {
        name = name.replace(` (${value})`, '').split(' · ')[0].trim();
    }
    if (!name || name === value) {
        // Kokoro ids read as "af_bella" → "Bella".
        const kokoro = /^[a-z]{2}_([a-z0-9]+)$/i.exec(value);
        name = kokoro ? kokoro[1].charAt(0).toUpperCase() + kokoro[1].slice(1) : '';
    }
    return name && name !== value ? name : '';
}

function castVoiceInfo(speaker, row = castAssignmentRow(speaker)) {
    const engine = castEngineName();
    const settings = typeof runtimeSettings !== 'undefined' ? runtimeSettings : null;
    const selections = typeof turboSelectionState !== 'undefined' ? turboSelectionState : {};
    const select = row?.querySelector('.voice-select');
    const sampleEngine = (typeof isPromptEngine === 'function' && isPromptEngine(engine))
        || (typeof isQwenCloneEngine === 'function' && isQwenCloneEngine(engine))
        || (typeof isOmniVoiceCloneEngine === 'function' && isOmniVoiceCloneEngine(engine));

    if (sampleEngine) {
        const refSelect = row?.querySelector('.reference-select');
        const reference = (refSelect?.value || (engine !== 'breeze_api' ? selections[speaker] : '') || '').trim();
        if (reference) {
            return { primary: castSampleName(reference), secondary: 'voice sample', state: 'set' };
        }
        if (engine === 'breeze_tts_2') {
            const profile = typeof findSpeakerProfile === 'function' ? findSpeakerProfile(speaker).profile : null;
            if (profile?.voice_design_prompt || profile?.voice) {
                return { primary: 'Designed from profile', secondary: '', state: 'default' };
            }
        }
        if (typeof isOmniVoiceCloneEngine === 'function' && isOmniVoiceCloneEngine(engine)) {
            const fallback = (typeof getGlobalReferenceSelection === 'function' ? getGlobalReferenceSelection() : '')
                || settings?.omnivoice_clone_default_prompt;
            if (fallback) return { primary: castSampleName(fallback), secondary: 'default sample', state: 'default' };
        }
        if (engine !== 'breeze_api') return { primary: 'No voice', secondary: '', state: 'missing' };
    }

    if (typeof isLocalAITtsEngine === 'function' && isLocalAITtsEngine(engine)) {
        const value = (row?.querySelector('.localai-voice-input')?.value || select?.value || '').trim();
        const prefix = typeof TTS_STORY_VOICE_REFERENCE_PREFIX !== 'undefined' ? TTS_STORY_VOICE_REFERENCE_PREFIX : 'tts-story://voice-prompts/';
        if (value.startsWith(prefix)) {
            return { primary: castSampleName(value.slice(prefix.length)), secondary: 'voice sample', state: 'set' };
        }
        if (value) return { primary: value, secondary: castCatalogDisplayName(select, value), state: 'set' };
        const fallback = settings?.localai_tts_default_voice || '';
        return fallback
            ? { primary: 'Default', secondary: fallback, state: 'default' }
            : { primary: 'No voice', secondary: '', state: 'missing' };
    }

    const value = select && !select.disabled ? (select.value || '').trim() : '';
    if (value) return { primary: value, secondary: castCatalogDisplayName(select, value), state: 'set' };
    if (window.isGeminiTtsEngine?.(engine)) {
        return { primary: 'Default', secondary: settings?.gemini_tts_default_voice || 'Kore', state: 'default' };
    }
    if (engine === 'breeze_api' && settings?.breeze_api_default_voice) {
        return { primary: 'Default', secondary: settings.breeze_api_default_voice, state: 'default' };
    }
    return { primary: 'No voice', secondary: '', state: 'missing' };
}

function castVoiceLabel(info) {
    return info.secondary ? `${info.primary} · ${info.secondary}` : info.primary;
}

/* ---------- entries + render ---------- */

function castEntries() {
    const readyState = typeof speakerReadyState !== 'undefined' ? speakerReadyState : {};
    // Index once per render instead of querying/scanning the whole cast for
    // every speaker. An open drawer's edited row overrides its hidden copy.
    const rows = new Map((typeof getAssignmentRows === 'function' ? getAssignmentRows() : [])
        .map(row => [row.dataset.speaker, row]));
    document.querySelectorAll('#speaker-edit-modal-body .voice-assignment-row')
        .forEach(row => rows.set(row.dataset.speaker, row));
    return castSpeakerNames.map(name => {
        const voice = castVoiceInfo(name, rows.get(name) || null);
        const fx = castFxValues(name);
        return {
            name,
            color: castSpeakerColor(name),
            ready: !!readyState[name],
            voice,
            voiceLabel: castVoiceLabel(voice),
            pitch: fx.pitch,
            speed: fx.speed,
            chunks: castChunkCount(name)
        };
    });
}

function castNextToReview(entries) {
    return entries.find(entry => entry.voice.state === 'missing') || entries.find(entry => !entry.ready) || null;
}

function castUpdateWorkflow(entries) {
    const missing = entries.filter(entry => entry.voice.state === 'missing').length;
    const ready = entries.filter(entry => entry.ready && entry.voice.state !== 'missing').length;
    const summary = document.getElementById('cast-assignment-summary');
    if (summary) summary.textContent = `${ready}/${entries.length} ready · ${missing} need a voice`;
    const next = document.getElementById('cast-review-next');
    if (next) next.disabled = !castNextToReview(entries);
    const engineLink = document.getElementById('cast-choose-engine');
    if (engineLink) engineLink.title = `Current engine: ${castEngineLabel() || 'not selected'}`;
}

/** Called by main.js displayStatistics after the per-speaker rows are built. */
function renderCastList(speakers, stats) {
    castSpeakerNames = Array.isArray(speakers) ? speakers.slice() : [];
    castStatsSnapshot = stats || null;
    castRenderNow({ assumeVisible: true });
}

/** Debounced re-render (voice / ready / FX / engine changes). */
function castViewRender() {
    if (castRenderQueued) return;
    castRenderQueued = true;
    const run = () => {
        castRenderQueued = false;
        castRenderNow();
    };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
    else setTimeout(run, 0);
}

function castMakeEl(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined && text !== null) el.textContent = text;
    return el;
}

function castIcon(name, extraClass = '') {
    const icon = castMakeEl('span', `ms${extraClass ? ` ${extraClass}` : ''}`, name);
    icon.setAttribute('aria-hidden', 'true');
    return icon;
}

// The whole row opens the drawer on click (main.js delegates on .speaker-tag);
// for keyboard and screen readers the name cell is the row's real button.
function castRowShell(entry, extraClass) {
    const row = castMakeEl('div', `speaker-tag cast-row ${extraClass}${entry.ready ? ' ready' : ''}`);
    row.dataset.speaker = entry.name;
    row.title = `Edit ${entry.name}`;
    row.style.setProperty('--cast-color', entry.color);
    const dot = castMakeEl('span', 'cast-dot');
    dot.setAttribute('aria-hidden', 'true');
    row.appendChild(dot);
    return row;
}

function castRowButton(entry, voiceText) {
    const text = castMakeEl('button', 'cast-row-text');
    text.type = 'button';
    text.setAttribute('aria-label', `Edit ${entry.name}: ${entry.voiceLabel}, ${entry.ready ? 'ready' : 'needs review'}`);
    text.appendChild(castMakeEl('span', 'cast-row-name', entry.name));
    const voice = castMakeEl('span', `cast-row-voice is-${entry.voice.state}`, voiceText);
    voice.title = entry.voiceLabel;
    text.appendChild(voice);
    return text;
}

function castBuildListRow(entry) {
    const row = castRowShell(entry, 'cast-list-row');
    row.appendChild(castRowButton(entry, entry.voiceLabel));

    if (entry.ready) {
        const status = castMakeEl('span', 'cast-row-status is-ready');
        status.append(castIcon('check'), 'Ready');
        row.appendChild(status);
    } else {
        row.appendChild(castMakeEl('span', 'cast-row-status is-review', 'Review'));
    }

    const play = castMakeEl('button', 'btn-circle cast-play');
    play.type = 'button';
    play.dataset.castAction = 'play';
    play.title = entry.voice.state === 'missing' ? 'Assign a voice to preview it' : `Play ${entry.name}'s voice`;
    play.setAttribute('aria-label', `Play ${entry.name}'s voice`);
    play.appendChild(castIcon('play_arrow'));
    row.appendChild(play);
    castApplyPlayState(play, castPreview?.speaker === entry.name ? castPreview.state : 'idle');
    return row;
}

function castBuildDetailRow(entry, showChunks) {
    const row = castRowShell(entry, 'cast-detail-row');
    row.appendChild(castRowButton(entry, entry.voice.primary));
    row.appendChild(castMakeEl('span', 'cast-num', typeof formatFxPitch === 'function' ? formatFxPitch(entry.pitch) : `${entry.pitch.toFixed(1)} st`));
    row.appendChild(castMakeEl('span', 'cast-num', typeof formatFxSpeed === 'function' ? formatFxSpeed(entry.speed) : `${entry.speed.toFixed(2)}×`));
    if (showChunks) row.appendChild(castMakeEl('span', 'cast-num cast-num-dim', String(entry.chunks ?? '–')));
    const status = castIcon(entry.ready ? 'check' : 'error', `cast-status-icon ${entry.ready ? 'is-ready' : 'is-review'}`);
    status.setAttribute('aria-hidden', 'true');
    status.title = entry.ready ? 'Ready' : 'Not marked ready';
    row.appendChild(status);
    return row;
}

function castBuildDetails(entries, stats) {
    const fragment = document.createDocumentFragment();
    const showChunks = entries.some(entry => entry.chunks !== null);
    const head = castMakeEl('div', `cast-details-head${showChunks ? '' : ' no-chunks'}`);
    head.setAttribute('aria-hidden', 'true');
    ['', 'Speaker', 'Pitch', 'Speed', ...(showChunks ? ['Chk'] : []), ''].forEach(label => {
        const cell = castMakeEl('span', '', label);
        if (label === 'Chk') cell.title = 'Chunks';
        head.appendChild(cell);
    });
    fragment.appendChild(head);
    entries.forEach(entry => {
        const row = castBuildDetailRow(entry, showChunks);
        if (!showChunks) row.classList.add('no-chunks');
        fragment.appendChild(row);
    });
    const totalChunks = Number(stats?.total_chunks) || 0;
    const total = castMakeEl('div', 'cast-details-total');
    total.appendChild(castMakeEl('span', '', `${totalChunks} ${totalChunks === 1 ? 'chunk' : 'chunks'}`));
    total.appendChild(castMakeEl('span', '', `≈ ${castFormatDuration(stats?.estimated_duration)} total`));
    fragment.appendChild(total);
    return fragment;
}

function castRenderNow(options = {}) {
    const list = document.getElementById('speakers-list');
    const section = document.getElementById('cast-section');
    if (!list) return;
    const statsSection = document.getElementById('stats-section');
    const stats = castCurrentStats();
    // main.js shows a "Speaker 1" placeholder row even for an empty manuscript;
    // until there is text, the Cast shows its empty state instead.
    const emptyManuscript = !!stats && !stats.has_speaker_tags && !(Number(stats.word_count) > 0);
    const active = castSpeakerNames.length > 0 && !emptyManuscript
        && (options.assumeVisible || !statsSection || statsSection.style.display !== 'none');
    const entries = castEntries();
    const emotionSegments = stats?.has_emotion_tags ? Number(stats.segments_with_emotion) || 0 : 0;
    const details = castViewMode === 'details';
    const signature = JSON.stringify([
        active, castViewMode, castEngineName(), emotionSegments, stats?.total_chunks, stats?.estimated_duration,
        entries.map(entry => [entry.name, entry.color, entry.ready, entry.voiceLabel, entry.voice.state,
            details ? entry.pitch : 0, details ? entry.speed : 0, entry.chunks])
    ]);
    castSyncDrawer();
    if (signature === castLastSignature && list.childElementCount) return;
    castLastSignature = signature;
    castUpdateWorkflow(active ? entries : []);

    section?.classList.toggle('has-cast', active);
    const count = section?.querySelector('.cast-count');
    if (count) count.textContent = active ? String(entries.length) : '';
    section?.querySelectorAll('.cast-view-toggle [data-cast-view]').forEach(button => {
        const on = button.dataset.castView === castViewMode;
        button.classList.toggle('is-active', on);
        button.setAttribute('aria-pressed', on ? 'true' : 'false');
    });

    const focusedSpeaker = list.contains(document.activeElement)
        ? document.activeElement.closest('.cast-row')?.dataset.speaker
        : null;
    const focusedPlay = !!focusedSpeaker && document.activeElement.matches('[data-cast-action="play"]');
    list.dataset.castView = castViewMode;
    list.replaceChildren();
    if (details) {
        list.appendChild(castBuildDetails(entries, stats));
    } else {
        entries.forEach(entry => list.appendChild(castBuildListRow(entry)));
    }
    if (emotionSegments) {
        const note = castMakeEl('p', 'cast-footnote cast-emotion-note');
        const segments = `${emotionSegments} ${emotionSegments === 1 ? 'segment' : 'segments'}`;
        note.append(castIcon('theater_comedy'), `Emotion tags in ${segments} are sent as Qwen3 instructions.`);
        list.appendChild(note);
    }
    if (focusedPlay) castPlayButtonFor(focusedSpeaker)?.focus({ preventScroll: true });
    else if (focusedSpeaker) castFocusRow(focusedSpeaker);

    document.dispatchEvent(new CustomEvent('cast:changed', {
        detail: {
            speakers: active
                ? entries.map(entry => ({ name: entry.name, color: entry.color, ready: entry.ready, voiceLabel: entry.voiceLabel }))
                : []
        }
    }));
}

function castFocusRow(speaker) {
    const row = Array.from(document.querySelectorAll('#speakers-list .cast-row')).find(item => item.dataset.speaker === speaker);
    const target = row?.querySelector('.cast-row-text');
    if (!target) return false;
    target.focus({ preventScroll: true });
    return true;
}

/* ---------- list play button: reuses the row's own preview controls ---------- */

function castPreviewSource(row) {
    const turbo = row.querySelector('[data-role="turbo-control"]');
    const sampleButton = row.querySelector('[data-role="voice-sample-preview-btn"]');
    if (turbo && !turbo.hidden && sampleButton && !sampleButton.disabled) return sampleButton;
    const catalog = row.querySelector('[data-role="kokoro-control"]');
    const selected = row.querySelector('[data-role="gemini-selected-preview"]');
    if (catalog && !catalog.hidden && selected && !selected.hidden && !selected.disabled) return selected;
    return row.querySelector('[data-role="fx-preview-btn"]');
}

function castSourceState(source) {
    if (!source) return 'idle';
    if (source.classList.contains('is-loading')) return 'loading';
    if (source.classList.contains('is-playing') || /^stop$/i.test(source.textContent.trim())) return 'playing';
    if (source.dataset.role === 'fx-preview-btn' && source.disabled) return 'loading';
    return 'idle';
}

function castApplyPlayState(button, state) {
    if (!button) return;
    button.classList.toggle('is-loading', state === 'loading');
    button.classList.toggle('is-playing', state === 'playing');
    const icon = button.querySelector('.ms');
    if (icon) icon.textContent = state === 'playing' ? 'stop' : state === 'loading' ? 'progress_activity' : 'play_arrow';
    button.setAttribute('aria-pressed', state === 'idle' ? 'false' : 'true');
}

function castPlayButtonFor(speaker) {
    return Array.from(document.querySelectorAll('#speakers-list .cast-row'))
        .find(row => row.dataset.speaker === speaker)?.querySelector('[data-cast-action="play"]') || null;
}

function castStopWatching() {
    if (!castPreview) return;
    castPreview.observer?.disconnect();
    const button = castPlayButtonFor(castPreview.speaker);
    castPreview = null;
    castApplyPlayState(button, 'idle');
}

function castUpdatePreviewState() {
    if (!castPreview) return;
    const state = castSourceState(castPreview.source);
    if (state !== 'idle') {
        castPreview.state = state;
        castPreview.seenActive = true;
        castApplyPlayState(castPlayButtonFor(castPreview.speaker), state);
        return;
    }
    if (!castPreview.seenActive) return;
    // Back to idle: surface errors the hidden row reported (Quick-test fallback).
    const statusText = castPreview.status?.textContent?.trim() || '';
    if (statusText && !/^(Rendering|Playing) preview/i.test(statusText) && typeof showNotification === 'function') {
        showNotification(`${castPreview.speaker}: ${statusText}`, 'warning');
    }
    castStopWatching();
}

function castPlay(speaker) {
    const row = castAssignmentRow(speaker);
    if (!row) return;
    if (castPreview && castPreview.speaker === speaker) {
        if (typeof stopSpeakerPreviewAudio === 'function') stopSpeakerPreviewAudio();
        castStopWatching();
        return;
    }
    castStopWatching();
    if (typeof stopSpeakerPreviewAudio === 'function') stopSpeakerPreviewAudio();
    if (castVoiceInfo(speaker).state === 'missing') {
        if (typeof showNotification === 'function') {
            showNotification(`Choose a voice for ${speaker} first. Click the speaker to open its settings.`, 'warning');
        }
        return;
    }
    if (typeof updateGeminiSelectedVoicePreview === 'function') updateGeminiSelectedVoicePreview(row);
    const source = castPreviewSource(row);
    if (!source || source.disabled) return;
    const observer = new MutationObserver(castUpdatePreviewState);
    observer.observe(source, { attributes: true, attributeFilter: ['class', 'disabled'], childList: true, characterData: true, subtree: true });
    castPreview = {
        speaker,
        source,
        observer,
        state: 'loading',
        seenActive: false,
        status: source.dataset.role === 'fx-preview-btn' ? row.querySelector('[data-role="fx-status"]') : null
    };
    source.click();
    castUpdatePreviewState();
    if (castPreview && !castPreview.seenActive) {
        // The control refused synchronously (e.g. Quick test without a voice).
        const statusText = castPreview.status?.textContent?.trim() || '';
        if (statusText && typeof showNotification === 'function') showNotification(`${speaker}: ${statusText}`, 'warning');
        castStopWatching();
    }
}

/* ---------- drawer ---------- */

function castDrawerElements() {
    return {
        overlay: document.getElementById('speaker-edit-modal-overlay'),
        modal: document.getElementById('speaker-edit-modal')
    };
}

function castDrawerOpen() {
    const { overlay } = castDrawerElements();
    return !!overlay && !overlay.classList.contains('hidden');
}

function castSyncDrawer() {
    if (!castDrawerOpen()) return;
    const speaker = typeof activeSpeakerModal !== 'undefined' ? activeSpeakerModal : null;
    if (!speaker) return;
    const { modal } = castDrawerElements();
    const chunks = castChunkCount(speaker);
    const kicker = modal?.querySelector('.speaker-drawer-kicker');
    if (kicker) {
        kicker.textContent = chunks === null ? 'Speaker' : `Speaker · ${chunks} ${chunks === 1 ? 'chunk' : 'chunks'}`;
    }
    modal?.querySelector('.speaker-drawer-dot')?.style.setProperty('background', castSpeakerColor(speaker));
    modal?.style.setProperty('--cast-color', castSpeakerColor(speaker));
    const engineLabel = modal?.querySelector('[data-role="speaker-engine-label"]');
    if (engineLabel) engineLabel.textContent = castEngineLabel();
}

function castAutosize(textarea) {
    if (!textarea || (window.CSS && CSS.supports && CSS.supports('field-sizing', 'content'))) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight + 2, 320)}px`;
}

/** The drawer is aria-modal: the app behind it (sidebar + screens) is inert while
    it is open. Modals stacked over it live at body level, outside .app-shell. */
function castSetBackgroundInert(inert) {
    const shell = document.querySelector('.app-shell');
    if (shell) shell.inert = inert;
}

/** Tabbable controls of the drawer, in order (for the Tab wrap). */
function castDrawerFocusable(modal) {
    const selector = 'a[href], button, input, select, textarea, summary, audio[controls], video[controls], [tabindex]';
    return Array.from(modal.querySelectorAll(selector)).filter(element => {
        if (element.disabled || element.tabIndex < 0 || element.type === 'hidden') return false;
        return typeof element.checkVisibility === 'function'
            ? element.checkVisibility({ visibilityProperty: true })
            : element.getClientRects().length > 0;
    });
}

/** main.js openSpeakerEditModal → after the row moved into the drawer. */
function castDrawerOpened() {
    const { modal } = castDrawerElements();
    if (!castDrawerReturnFocus || !modal?.contains(castDrawerReturnFocus)) {
        castDrawerReturnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    }
    castSetBackgroundInert(true);
    castSyncDrawer();
    modal?.querySelector('.speaker-drawer-body')?.scrollTo?.(0, 0);
    castAutosize(modal?.querySelector('[data-role="speaker-profile-description"]'));
    if (modal && !modal.contains(document.activeElement)) {
        modal.setAttribute('tabindex', '-1');
        modal.focus({ preventScroll: true });
    }
}

/** main.js closeSpeakerEditModal → after the row went back to the hidden list. */
function castDrawerClosed(speaker) {
    castSetBackgroundInert(false);
    castRenderNow();
    const returnFocus = castDrawerReturnFocus;
    castDrawerReturnFocus = null;
    if (castFocusRow(speaker)) return;
    if (returnFocus && document.contains(returnFocus)) returnFocus.focus?.({ preventScroll: true });
}

/* ---------- wiring ---------- */

function castViewInit() {
    const section = document.getElementById('cast-section');
    const list = document.getElementById('speakers-list');
    if (!section || !list) return;

    document.getElementById('cast-choose-engine')?.addEventListener('click', event => {
        event.preventDefault();
        const engine = document.getElementById('job-tts-engine');
        engine?.scrollIntoView({behavior: 'smooth', block: 'nearest'});
        engine?.focus({preventScroll: true});
    });
    document.getElementById('cast-review-next')?.addEventListener('click', () => {
        const next = castNextToReview(castEntries());
        if (next && typeof openSpeakerEditModal === 'function') openSpeakerEditModal(next.name);
    });

    section.querySelector('.cast-view-toggle')?.addEventListener('click', event => {
        const button = event.target instanceof Element ? event.target.closest('[data-cast-view]') : null;
        if (!button || button.dataset.castView === castViewMode) return;
        castViewMode = button.dataset.castView === 'details' ? 'details' : 'list';
        castRenderNow();
    });

    list.addEventListener('click', event => {
        const play = event.target instanceof Element ? event.target.closest('[data-cast-action="play"]') : null;
        if (!play) return;
        event.preventDefault();
        const speaker = play.closest('.cast-row')?.dataset.speaker;
        if (speaker) castPlay(speaker);
    });

    // hideAnalysis() and resetVoiceAssignments() hide the cast body directly.
    const statsSection = document.getElementById('stats-section');
    if (statsSection && typeof MutationObserver === 'function') {
        new MutationObserver(castViewRender).observe(statsSection, { attributes: true, attributeFilter: ['style'] });
    }
    document.getElementById('job-tts-engine')?.addEventListener('change', castViewRender);
    window.addEventListener(window.CHATTERBOX_VOICES_EVENT || 'chatterboxVoices:updated', castViewRender);

    // The drawer: Esc closes it through closeSpeakerEditModal (which keeps the
    // Gemini busy guard), unless another modal is stacked above it.
    document.addEventListener('keydown', event => {
        if (event.key !== 'Escape' || event.defaultPrevented || !castDrawerOpen()) return;
        const { overlay } = castDrawerElements();
        const topmost = typeof findTopmostModalOverlay === 'function' ? findTopmostModalOverlay() : overlay;
        if (topmost !== overlay || document.querySelector('dialog[open]')) return;
        event.preventDefault();
        if (typeof closeSpeakerEditModal === 'function') closeSpeakerEditModal();
    });

    // Safety net: the app behind follows the overlay's visibility even if some
    // path hides it without closeSpeakerEditModal (castDrawerClosed).
    const drawerOverlay = castDrawerElements().overlay;
    if (drawerOverlay && typeof MutationObserver === 'function') {
        new MutationObserver(() => castSetBackgroundInert(castDrawerOpen()))
            .observe(drawerOverlay, { attributes: true, attributeFilter: ['class'] });
    }

    const drawer = document.getElementById('speaker-edit-modal');
    // Tab / Shift+Tab wrap between the drawer's first and last controls.
    drawer?.addEventListener('keydown', event => {
        if (event.key !== 'Tab' || event.defaultPrevented || !castDrawerOpen()) return;
        const focusable = castDrawerFocusable(drawer);
        const active = document.activeElement;
        if (!focusable.length) {
            event.preventDefault();
            drawer.focus({ preventScroll: true });
        } else if (event.shiftKey && (active === drawer || active === focusable[0])) {
            event.preventDefault();
            focusable[focusable.length - 1].focus();
        } else if (!event.shiftKey && active === focusable[focusable.length - 1]) {
            event.preventDefault();
            focusable[0].focus();
        }
    });
    // The drawer's help icon opens the Help Center tab (help.js, document
    // listener); close the drawer first so it does not cover the article.
    drawer?.addEventListener('click', event => {
        if (event.target instanceof Element && event.target.closest('[data-help-id]')
            && typeof closeSpeakerEditModal === 'function') {
            closeSpeakerEditModal();
        }
    });
    drawer?.addEventListener('input', event => {
        if (event.target instanceof HTMLTextAreaElement && event.target.dataset.role === 'speaker-profile-description') {
            castAutosize(event.target);
        }
    });

    castRenderNow();
}

document.addEventListener('DOMContentLoaded', castViewInit);
