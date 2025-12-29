// Voice Manager - Handles voice library and selection

let voiceData = null;
// Make availableVoices globally accessible for main.js
window.availableVoices = null;
window.customVoiceMap = window.customVoiceMap || {};

const audioPreviewCache = {};
let currentPreviewAudio = null;
let currentPreviewItem = null;
let samplesReady = false;
let lastFailedSamples = [];
let customVoices = [];

const generateSamplesBtnId = 'generate-voice-samples-btn';
const regenerateSamplesBtnId = 'regenerate-voice-samples-btn';
const sampleStatusId = 'voice-sample-status';

const VOICES_UPDATED_EVENT = window.VOICES_UPDATED_EVENT || 'voices:updated';
window.VOICES_UPDATED_EVENT = VOICES_UPDATED_EVENT;

// Load voices on page load
document.addEventListener('DOMContentLoaded', () => {
    loadVoices();
    loadCustomVoices();
    setupCustomVoiceModal();
});

// Load available voices from API
async function loadVoices() {
    try {
        const response = await fetch('/api/voices');
        const data = await response.json();
        
        if (data.success) {
            voiceData = data.voices;
            window.availableVoices = data.voices; // Make globally accessible
            updateCustomVoiceMap(data.voices);
            samplesReady = data.samples_ready;
            updateSampleStatus(data);
            displayVoiceLibrary(data.voices);
            emitVoicesUpdated();
        } else {
            displaySampleError(data.error || 'Unable to load voices');
        }
    } catch (error) {
        console.error('Error loading voices:', error);
        displaySampleError('Error loading voices');
    }
}

function emitVoicesUpdated() {
    const detail = {
        voices: voiceData,
        samplesReady,
        customVoiceMap: window.customVoiceMap,
    };
    window.dispatchEvent(new CustomEvent(VOICES_UPDATED_EVENT, { detail }));
}

function updateCustomVoiceMap(voices) {
    const map = {};
    if (voices && typeof voices === 'object') {
        Object.values(voices).forEach(config => {
            if (!config) return;
            const langCode = config.lang_code;
            (config.custom_voices || []).forEach(entry => {
                if (!entry || !entry.code) return;
                map[entry.code] = {
                    ...entry,
                    lang_code: entry.lang_code || langCode,
                };
            });
        });
    }
    window.customVoiceMap = map;
}

async function loadCustomVoices() {
    try {
        const response = await fetch('/api/custom-voices');
        const data = await response.json();
        if (data.success) {
            customVoices = data.voices || [];
            renderCustomVoices();
        } else {
            showToast(data.error || 'Unable to load custom voices', 'error');
        }
    } catch (error) {
        console.error('Failed to load custom voices', error);
        showToast('Failed to load custom voices', 'error');
    }
}

function renderCustomVoices() {
    const container = document.getElementById('custom-voices-list');
    if (!container) return;

    if (!customVoices.length) {
        container.innerHTML = '<p class="help-text">You haven’t created any blends yet. Click “New Custom Voice” to get started.</p>';
        return;
    }

    container.innerHTML = '';
    customVoices.forEach(entry => {
        const card = document.createElement('div');
        card.className = 'custom-voice-card';
        card.dataset.voiceCode = entry.code;
        card.innerHTML = `
            <div class="custom-voice-title">
                <div>
                    <strong>${entry.name}</strong>
                </div>
                <span class="voice-badge">Lang ${entry.lang_code?.toUpperCase() ?? ''}</span>
            </div>
            <div class="custom-voice-components">
                ${renderComponentList(entry.components)}
            </div>
            <div class="custom-voice-meta">
                <span class="badge-muted">${entry.code}</span>
                ${entry.updated_at ? `<span>Updated ${new Date(entry.updated_at).toLocaleString()}</span>` : ''}
            </div>
            <div class="custom-voice-actions">
                <button class="btn-ghost" data-action="edit">Edit</button>
                <button class="btn-danger" data-action="delete">Delete</button>
            </div>
        `;

        card.querySelector('[data-action="edit"]').addEventListener('click', () => openCustomVoiceModal(entry));
        card.querySelector('[data-action="delete"]').addEventListener('click', () => deleteCustomVoice(entry));
        container.appendChild(card);
    });
}

function renderComponentList(components = []) {
    if (!components.length) {
        return '<em>No components defined.</em>';
    }
    return components.map(comp => {
        const weight = Number(comp.weight ?? 1).toFixed(2).replace(/\.00$/, '');
        return `<div>${comp.voice} <span class="badge-muted">x${weight}</span></div>`;
    }).join('');
}

function setupCustomVoiceModal() {
    const overlay = document.getElementById('custom-voice-modal-overlay');
    const modal = document.getElementById('custom-voice-modal');
    const openBtn = document.getElementById('create-custom-voice-btn');
    const closeBtn = document.getElementById('custom-voice-modal-close');
    const cancelBtn = document.getElementById('custom-voice-cancel');
    const saveBtn = document.getElementById('custom-voice-save');
    const addComponentBtn = document.getElementById('add-component-btn');

    if (!overlay || !modal) return;

    function closeModal() {
        overlay.classList.add('hidden');
        modal.classList.add('hidden');
        modal.dataset.editCode = '';
        document.getElementById('custom-voice-form').reset();
        const rows = document.getElementById('custom-voice-components');
        rows.innerHTML = '';
    }

    function openModal(entry = null) {
        overlay.classList.remove('hidden');
        modal.classList.remove('hidden');
        const title = document.getElementById('custom-voice-modal-title');
        const nameInput = document.getElementById('custom-voice-name');
        const langSelect = document.getElementById('custom-voice-lang');
        const notesInput = document.getElementById('custom-voice-notes');
        const rows = document.getElementById('custom-voice-components');

        rows.innerHTML = '';
        if (entry) {
            modal.dataset.editCode = entry.code;
            title.textContent = 'Edit Custom Voice';
            nameInput.value = entry.name || '';
            langSelect.value = entry.lang_code || 'a';
            notesInput.value = entry.notes || '';
            (entry.components || []).forEach(component => addComponentRow(component));
        } else {
            modal.dataset.editCode = '';
            title.textContent = 'Create Custom Voice';
            nameInput.value = '';
            langSelect.value = 'a';
            notesInput.value = '';
            addComponentRow();
        }
    }

    function addComponentRow(component = null) {
        const rows = document.getElementById('custom-voice-components');
        const row = document.createElement('div');
        row.className = 'component-row';
        const voiceSelect = buildVoiceSelect(component?.voice, document.getElementById('custom-voice-lang').value);
        voiceSelect.classList.add('component-voice-select');
        const weightInput = document.createElement('input');
        weightInput.type = 'number';
        weightInput.min = '0.1';
        weightInput.step = '0.1';
        weightInput.value = component?.weight ?? 1;
        weightInput.classList.add('component-weight-input');
        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'remove-component';
        removeBtn.innerHTML = '&times;';
        removeBtn.addEventListener('click', () => {
            if (rows.children.length > 1) {
                row.remove();
            } else {
                showToast('Need at least one component.', 'warning');
            }
        });
        row.appendChild(voiceSelect);
        row.appendChild(weightInput);
        row.appendChild(removeBtn);
        rows.appendChild(row);
    }

    function buildVoiceSelect(selectedVoice = '', langCode = 'a') {
        const select = document.createElement('select');
        const voices = getVoicesForLang(langCode);
        voices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice;
            option.textContent = voice;
            if (voice === selectedVoice) {
                option.selected = true;
            }
            select.appendChild(option);
        });
        return select;
    }

    function getVoicesForLang(langCode = 'a') {
        if (!window.availableVoices) return [];
        const entry = Object.values(window.availableVoices).find(cfg => cfg.lang_code === langCode);
        return entry ? entry.voices : [];
    }

    openBtn?.addEventListener('click', () => openModal());
    closeBtn?.addEventListener('click', closeModal);
    cancelBtn?.addEventListener('click', closeModal);
    overlay?.addEventListener('click', event => {
        if (event.target === overlay) closeModal();
    });
    addComponentBtn?.addEventListener('click', () => addComponentRow());

    document.getElementById('custom-voice-lang')?.addEventListener('change', event => {
        const rows = document.querySelectorAll('.component-row select');
        rows.forEach(select => {
            const value = select.value;
            const newSelect = buildVoiceSelect(value, event.target.value);
            newSelect.className = select.className;
            select.replaceWith(newSelect);
        });
    });

    saveBtn?.addEventListener('click', async () => {
        const form = document.getElementById('custom-voice-form');
        const name = document.getElementById('custom-voice-name').value.trim();
        const lang = document.getElementById('custom-voice-lang').value;
        const notes = document.getElementById('custom-voice-notes').value.trim();
        const components = Array.from(document.querySelectorAll('.component-row')).map(row => ({
            voice: row.querySelector('select').value,
            weight: parseFloat(row.querySelector('.component-weight-input').value || 1),
        }));

        if (!name) {
            showToast('Name is required.', 'warning');
            return;
        }
        if (!components.length) {
            showToast('Add at least one component.', 'warning');
            return;
        }

        const payload = { name, lang_code: lang, notes, components };
        const isEdit = Boolean(modal.dataset.editCode);
        const url = isEdit ? `/api/custom-voices/${modal.dataset.editCode}` : '/api/custom-voices';
        const method = isEdit ? 'PUT' : 'POST';

        try {
            const response = await fetch(url, {
                method,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || 'Failed to save custom voice');
            }
            showToast(isEdit ? 'Custom voice updated.' : 'Custom voice created!', 'success');
            closeModal();
            await loadCustomVoices();
            await loadVoices();
        } catch (error) {
            console.error('Failed to save custom voice', error);
            showToast(error.message || 'Failed to save custom voice.', 'error');
        }
    });

    function openCustomVoiceModal(entry) {
        openModal(entry);
    }

    function deleteCustomVoice(entry) {
        if (!entry) return;
        if (!confirm(`Delete custom voice "${entry.name}"? This cannot be undone.`)) {
            return;
        }
        fetch(`/api/custom-voices/${entry.code}`, {
            method: 'DELETE',
        })
            .then(res => res.json())
            .then(data => {
                if (!data.success) {
                    throw new Error(data.error || 'Failed to delete custom voice');
                }
                showToast('Custom voice deleted.', 'success');
                loadCustomVoices();
                loadVoices();
            })
            .catch(err => {
                console.error('Delete custom voice failed', err);
                showToast(err.message || 'Failed to delete custom voice', 'error');
            });
    }

    window.openCustomVoiceModal = openCustomVoiceModal;
    window.deleteCustomVoice = deleteCustomVoice;
    window.addComponentRow = addComponentRow;
}

function summarizeVoiceList(list, max = 5) {
    if (!Array.isArray(list) || list.length === 0) {
        return 'None';
    }
    const uniqueVoices = [...new Set(list)];
    const shown = uniqueVoices.slice(0, max);
    const remainder = uniqueVoices.length - shown.length;
    return remainder > 0
        ? `${shown.join(', ')} +${remainder} more`
        : shown.join(', ');
}

function updateSampleStatus(data) {
    const statusContainer = document.getElementById(sampleStatusId);
    const buttonContainer = document.getElementById('voice-sample-controls');

    if (!statusContainer || !buttonContainer) {
        return;
    }

    const failedList = Array.isArray(data.failed)
        ? data.failed
        : lastFailedSamples;
    if (Array.isArray(data.failed)) {
        lastFailedSamples = data.failed;
    }

    const missingList = Array.isArray(data.missing_samples) ? data.missing_samples : [];
    const missingCount = missingList.length;
    const failedCount = failedList.length;
    const totalVoices = data.total_unique_voices || 0;
    const generatedCount = data.sample_count || 0;

    let summaryMessage = '';
    let statusClass = 'info';
    const detailMessages = [];

    if (missingCount === 0 && failedCount === 0 && generatedCount > 0) {
        summaryMessage = `All ${generatedCount} voice previews are ready.`;
        statusClass = 'success';
        buttonContainer.style.display = 'none';
    } else {
        const missingSummary = missingCount > 0
            ? `${missingCount} of ${totalVoices} voices still need previews`
            : 'Some voices are ready to preview';
        summaryMessage = missingSummary;

        if (missingCount > 0) {
            detailMessages.push(`Missing previews: ${summarizeVoiceList(missingList)}`);
        }

        if (failedCount > 0) {
            const failedNames = failedList
                .map(item => (typeof item === 'string' ? item : item.voice))
                .filter(Boolean);
            detailMessages.push(`Failed to generate: ${summarizeVoiceList(failedNames)}`);
            statusClass = 'warning';
        } else if (missingCount > 0) {
            statusClass = 'warning';
        } else {
            statusClass = 'info';
        }

        buttonContainer.style.display = 'flex';
    }

    statusContainer.className = `sample-status ${statusClass}`.trim();
    statusContainer.innerHTML = `
        <div>${summaryMessage}</div>
        ${detailMessages.length ? `<div class="sample-status-details">${detailMessages.join('<br>')}</div>` : ''}
    `;
}

function displaySampleError(message) {
    const statusContainer = document.getElementById(sampleStatusId);
    const buttonContainer = document.getElementById('voice-sample-controls');
    if (statusContainer) {
        statusContainer.textContent = message;
        statusContainer.className = 'sample-status error';
    }
    if (buttonContainer) {
        buttonContainer.style.display = 'flex';
    }
}

// Display voice library
function displayVoiceLibrary(voices) {
    const container = document.getElementById('voice-library');
    container.innerHTML = '';
    
    const languageNames = {
        'american_english': '🇺🇸 American English',
        'british_english': '🇬🇧 British English',
        'spanish': '🇪🇸 Spanish',
        'french': '🇫🇷 French',
        'hindi': '🇮🇳 Hindi',
        'japanese': '🇯🇵 Japanese',
        'chinese': '🇨🇳 Chinese',
        'brazilian_portuguese': '🇧🇷 Brazilian Portuguese',
    };
    
    for (const [key, config] of Object.entries(voices)) {
        const category = document.createElement('div');
        category.className = 'voice-category';
        
        const title = document.createElement('h3');
        title.textContent = languageNames[key] || key;
        category.appendChild(title);
        
        const list = document.createElement('ul');
        list.className = 'voice-list';
        
        config.voices.forEach(voice => {
            const item = document.createElement('li');
            item.className = 'voice-item';
            item.dataset.voice = voice;
            item.dataset.langCode = config.lang_code;
            
            const samplePath = (config.samples && config.samples[voice]) || null;
            if (samplePath) {
                item.classList.add('has-preview');
                item.dataset.samplePath = samplePath;
            }
            
            const friendlyName = voice.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            
            const info = document.createElement('div');
            info.className = 'voice-info';
            info.innerHTML = `
                <span class="voice-name">${friendlyName}</span>
                <span class="voice-code">${voice}</span>
            `;
            item.appendChild(info);
            
            const status = document.createElement('div');
            status.className = 'voice-status';
            status.textContent = samplePath ? 'Preview ready' : 'Preview unavailable';
            if (!samplePath) {
                status.classList.add('muted');
            }
            item.appendChild(status);
            
            item.addEventListener('click', () => {
                playVoicePreview(voice, config.lang_code, samplePath, item);
            });
            
            list.appendChild(item);
        });
        
        category.appendChild(list);
        container.appendChild(category);
    }
}

async function generateSamples(overwrite = false) {
    const button = document.getElementById(generateSamplesBtnId);
    const regenButton = document.getElementById(regenerateSamplesBtnId);
    const statusContainer = document.getElementById(sampleStatusId);

    if (!button || !statusContainer) {
        return;
    }

    button.disabled = true;
    button.textContent = overwrite ? 'Regenerating samples…' : 'Generating samples…';
    if (regenButton) {
        regenButton.disabled = true;
    }
    statusContainer.textContent = 'Generating preview samples, please wait…';
    statusContainer.className = 'sample-status info';

    try {
        const response = await fetch('/api/voices/samples', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ overwrite })
        });

        const data = await response.json();

        if (data.success) {
            samplesReady = data.samples_ready;
            lastFailedSamples = Array.isArray(data.failed) ? data.failed : [];
            if (data.voices) {
                voiceData = data.voices;
                window.availableVoices = data.voices;
                updateCustomVoiceMap(data.voices);
                updateSampleStatus(data);
                displayVoiceLibrary(data.voices);
                emitVoicesUpdated();
            } else {
                await loadVoices();
            }
            const failedCount = data.failed ? data.failed.length : 0;
            const generatedCount = data.generated ? data.generated.length : 0;
            if (failedCount > 0) {
                showToast(`${generatedCount} previews generated, ${failedCount} failed. Check status for details.`, 'info');
            } else {
                showToast('Voice previews generated successfully!', 'success');
            }
        } else {
            throw new Error(data.error || 'Unknown error');
        }
    } catch (error) {
        console.error('Error generating voice samples:', error);
        showToast('Failed to generate voice samples: ' + error.message, 'error');
        displaySampleError('Failed to generate voice samples. Please check the server logs.');
    } finally {
        const refreshedButton = document.getElementById(generateSamplesBtnId);
        const refreshedRegenButton = document.getElementById(regenerateSamplesBtnId);
        if (refreshedButton) {
            refreshedButton.disabled = false;
            refreshedButton.textContent = 'Generate Voice Previews';
        }
        if (refreshedRegenButton) {
            refreshedRegenButton.disabled = false;
        }
    }
}

function showToast(message, type) {
    if (window.showNotification) {
        window.showNotification(message, type);
    } else {
        alert(message);
    }
}

// Play voice preview using generated samples
function playVoicePreview(voice, langCode, samplePath, listItem) {
    if (!samplePath) {
        alert(`Preview not available for ${voice}. Click "Generate Voice Previews" to create samples.`);
        return;
    }

    if (currentPreviewAudio) {
        currentPreviewAudio.pause();
        currentPreviewAudio.currentTime = 0;
        if (currentPreviewItem) {
            currentPreviewItem.classList.remove('playing');
        }
    }

    if (!audioPreviewCache[voice]) {
        audioPreviewCache[voice] = new Audio(samplePath);
    }

    const audio = audioPreviewCache[voice];
    currentPreviewAudio = audio;
    currentPreviewItem = listItem;

    audio.currentTime = 0;
    audio.play().then(() => {
        listItem.classList.add('playing');
    }).catch(err => {
        console.error('Error playing preview:', err);
        alert('Unable to play preview. See console for details.');
    });

    audio.onended = () => {
        listItem.classList.remove('playing');
        currentPreviewAudio = null;
        currentPreviewItem = null;
    };
}

// Get voice info
function getVoiceInfo(voiceName) {
    if (!voiceData) return null;
    
    for (const [key, config] of Object.entries(voiceData)) {
        if (config.voices.includes(voiceName)) {
            return {
                language: key,
                lang_code: config.lang_code,
                voice: voiceName
            };
        }
    }
    
    return null;
}
