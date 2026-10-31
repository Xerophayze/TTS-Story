/* Guided first-run tour. Navigation is automatic; setup and generation are user actions. */
(() => {
    const storageKey = 'tts-story-guided-tutorial-step';
    const sampleStory = `[narrator]Chapter 1: The Missing Light[/narrator]

[narrator]Mira found the little lighthouse dark on the night of the village festival. At its door, her friend Finn held up an empty lantern.[/narrator]

[mira-female]The ships need that light. Where did the lantern go?[/mira-female]

[finn-male]I saw a glow near the old garden. Let's look together.[/finn-male]

[narrator]They followed a trail of golden sparks through the grass.[/narrator]

[narrator]Chapter 2: A Light to Share[/narrator]

[narrator]In the garden, a tiny fox curled beside the missing lantern. Its warm light had kept her safe from the cold.[/narrator]

[mira-female]You needed it too, didn't you?[/mira-female]

[finn-male]We can bring a blanket and leave her a smaller light.[/finn-male]

[narrator]Together they made the fox a cozy bed, then carried the lantern back to the lighthouse. Its beam swept across the harbor, and the festival boats answered with a chorus of cheerful bells.[/narrator]`;
    const firstAudioSteps = [
        { title: 'Choose your setup', target: '[data-settings-section="engine-settings-group"]', page: 'settings', section: 'engine-settings-group', text: 'Here you can install local engines and configure cloud providers. The tour highlights controls; you decide what to install or connect.' },
        { title: 'Choose a local engine', target: '[data-engine-tab="kokoro"]', page: 'settings', section: 'engine-settings-group', engine: 'kokoro', text: 'Kokoro is a lightweight starting point with built-in voices. Its settings panel is open; you can choose another local engine instead.' },
        { title: 'Install and wait', target: '#engine-panel-kokoro [data-install-engine], #engine-panel-kokoro', page: 'settings', section: 'engine-settings-group', engine: 'kokoro', text: 'If Kokoro is not installed, click Install engine here and wait for it to finish. Follow any restart prompt, then return to the tutorial. If it is already ready, continue.' },
        { title: 'Connect a cloud engine (optional)', target: '#remote-engine-tabs-label', page: 'settings', section: 'engine-settings-group', text: 'Cloud engines appear here. Select your provider, enter its API key in its settings, and save. Cloud services may charge for usage; a local engine does not need a cloud key.' },
        { title: 'Set up text preparation (optional)', target: '#gemini-settings-group', page: 'settings', section: 'gemini-settings-group', text: 'Prep text uses an optional language model to help tag speakers. Configure a provider and its credentials here if you want to use it, then save settings. You can also enter plain or tagged text yourself.' },
        { title: 'Save your settings', target: '#save-settings-btn', page: 'settings', text: 'If you entered or changed provider settings, select Save settings. Keep API keys private. You can return here later to add more engines.' },
        { title: 'Add your manuscript', target: '#input-text', page: 'generate', sample: true, text: 'Paste or type a short passage here, or choose Use sample text for a short two-chapter story with three speakers. The sample is already tagged, so Prep text is optional.' },
        { title: 'Select a narration engine', target: '#job-tts-engine', page: 'generate', text: 'Choose an installed local engine or a configured cloud provider. The available voice choices depend on the engine.' },
        { title: 'Cast your speakers', target: '#cast-section', page: 'generate', text: 'Click each speaker in Cast to open their properties, choose and preview a compatible voice, then mark them Ready. Close the speaker drawer before continuing.' },
        { title: 'Generate your first audio', target: '#generate-btn', page: 'generate', text: 'Once each speaker has a voice, click Generate audio. Cloud generation may be billable. The first local run may download additional model files.' },
        { title: 'Follow your job', target: '#queue-list', page: 'queue', text: 'Your job appears in the Job Queue. Wait until it finishes before continuing; the first local run may take longer while model files download.' },
        { title: 'Open the Library', target: '.tab-button[data-tab="library"]', text: 'Once your job is complete, click Library to find your finished audio. You can also click Next to open it.' },
        { title: 'Listen to your audio', target: '#library-items .library-item:first-child', fallback: '#library-list', page: 'library', library: true, text: 'The first visible production is highlighted. Click its play button to listen, or open its chapters to review and download them. If this is an older job, find your new production by title. You can replay this tour from Guided Tours.' }
    ];
    const tours = {
        first: firstAudioSteps,
        settings: [
            {title:'Settings overview', page:'settings', section:'quick-settings-group', target:'[data-settings-section="quick-settings-group"]', text:'Settings stores application preferences locally. Use the sections on the left to configure engines, audio, text preparation, and remote administration.'},
            {title:'Engine categories', page:'settings', section:'engine-settings-group', target:'#local-engine-tabs-label', text:'Local engines run on this computer in isolated environments. Green indicates ready; grey indicates not installed. Select an engine to see its requirements.'},
            {title:'Install and monitor', page:'settings', section:'engine-settings-group', engine:'kokoro', target:'#engine-panel-kokoro', text:'The engine panel offers Install, Repair, or Remove as appropriate. Installation logs remain available after navigation. Wait for completion and follow any restart prompt; projects and voice samples are preserved when an engine is removed.'},
            {title:'Cloud and remote engines', page:'settings', section:'engine-settings-group', target:'#remote-engine-tabs-label', text:'Cloud services and self-hosted servers appear separately. Select a provider to configure its key, endpoint, model, or region. Cloud requests can incur charges.'},
            {title:'Optional transcription tool', page:'settings', section:'engine-settings-group', target:'#utility-tabs-label', text:'Voice transcription is a separate optional CPU utility. It can help produce transcripts for reference clips without installing its dependencies into another engine.'},
            {title:'Audio and generation defaults', page:'settings', section:'audio-settings-group', target:'#audio-settings-group', text:'Set chunk size, crossfade, pauses, silence, speed, and supported concurrency. Smaller chunks can help memory, but may affect phrasing. Test changes on a short passage.'},
            {title:'Text-prep provider', page:'settings', section:'gemini-settings-group', target:'#gemini-settings-group', text:'Choose a cloud or local language-model provider for optional Prep text and profile building. Cloud credentials here are separate from speech-provider credentials when the engine requires its own key.'},
            {title:'Save settings', page:'settings', target:'#save-settings-btn', text:'Save your changes here. Some provider panels also have their own save or test buttons. Engine installation and backend restarts run only when you choose them.'}
        ],
        voices: [
            {title:'Explore Voices', page:'voices', subtab:'prompts', target:'#voices-subtab-prompts', text:'Voice prompts are reusable reference recordings for compatible cloning engines. A preset voice, a local recording, and a hosted designed voice are different kinds of voice.'},
            {title:'Find and organize samples', page:'voices', subtab:'prompts', target:'#voice-list-controls', text:'Search by name and filter by source, gender, or language. These labels help organize recordings; they do not change how a voice sounds.'},
            {title:'Add your own recording', page:'voices', subtab:'prompts', target:'#voice-add-btn', text:'Use Add voice to import a clean recording and metadata, or drop supported audio files onto the upload area. Use one speaker and include an exact transcript when the engine requires it.'},
            {title:'External voices', page:'voices', subtab:'prompts', target:'#load-external-voices-btn', text:'Load external voices retrieves a GitHub-hosted catalog over the internet. Previewing an entry does not save it: use its Download action to add a local copy for compatible engines. Review the source and rights before use.'},
            {title:'Preview and manage', page:'voices', subtab:'prompts', target:'#chatterbox-voice-list', fallback:'#voice-list-controls', text:'Use each row to play, edit metadata, or open more actions. Archive hides a sample without deleting it; deleting can break projects that reference it. Transcripts can be entered or generated with the optional local utility.'},
            {title:'Kokoro presets', page:'voices', subtab:'kokoro', target:'#kokoro-voices-section', text:'Kokoro voices are built-in presets grouped by language. Click a voice to hear a preview. These are selected as Kokoro voice codes, not reference recordings for every engine.'},
            {title:'Kokoro blends', page:'voices', subtab:'custom', target:'#custom-voices-section', text:'Create and manage combinations of compatible Kokoro voices here. A blend is for Kokoro rather than a generic reference sample.'},
            {title:'Design a voice', page:'voices', subtab:'qwen', target:'#qwen-voices-section', text:'Voice creation can generate candidates using supported design engines. Review candidates and save a chosen local sample; availability and licensing depend on the engine.'},
            {title:'Hosted Gemini voices', page:'voices', subtab:'gemini-hosted', target:'#gemini-hosted-voices-section', text:'Gemini custom voices are hosted by Google and managed separately from local recordings. Creation and synthesis can be billable; deleting a hosted voice can affect future jobs using it.'},
            {title:'Assign a voice', page:'generate', target:'#cast-section', text:'Choose your narration engine on Generate, open a speaker in Cast, and select a compatible preset, sample, or hosted voice. Preview it and save the project.'}
        ],
        manuscript: [
            {title:'Your manuscript workspace', page:'generate', target:'#text-input-wrapper', text:'The manuscript is the source for speaker analysis and synthesis. Keep an external copy of important work and save project snapshots before making substantial changes.'},
            {title:'Try the examples', page:'generate', target:'#text-input-wrapper', examples:true, text:'Insert the raw example to practice Prep text, or insert the already-tagged version to inspect speaker tags. You can also download either text file and drag it onto the editor. Insertion replaces current editor text only after confirmation.'},
            {title:'Import documents', page:'generate', target:'#browse-document-btn', text:'Use Load document or drag a supported TXT, PDF, Word, RTF, EPUB, ODT, Markdown, or HTML file into the manuscript. Multiple files are appended in order; review extracted text and keep your originals.'},
            {title:'Choose a prep prompt', page:'generate', target:'#gemini-preset-select', text:'Choose the instructions that suit your task. For source-exact books, select an appropriate strict book conversion prompt and verify that its wording matches your needs.'},
            {title:'Prepare text with an LLM', page:'generate', target:'#gemini-process-btn', text:'Prep text sends the manuscript to your configured language model and can add speaker tags or directions. Cloud use may be billable. When preparation completes it replaces the manuscript, so keep the original and review the result carefully.'},
            {title:'Raw tags and Script view', page:'generate', target:'[data-manuscript-view="raw"]', text:'Raw tags lets you edit matching speaker bookends. Script view presents the same text as readable speaker passages; switch views to inspect the result. A direction is guidance, not spoken dialogue.'},
            {title:'Check tags and cast', page:'generate', target:'#cast-section', text:'Prep Text prepares the manuscript first, then builds character profiles and voice types. Those details may take longer to appear and should populate automatically; if any are missing, use Build profiles. Check speaker tags and choose a voice for each speaker.'},
            {title:'Review chapter splitting', page:'generate', target:'#chapters-section-title', text:'Enable separate chapter files and review detected section headings before submitting a long manuscript. A combined Full Story can also be created.'},
            {title:'Alternate pronunciations', page:'generate', target:'#alt-word-registry-btn', text:'The Alt word registry helps with words an engine mispronounces. It substitutes text during synthesis without changing the manuscript. Click Next to open the registry.'},
            {title:'Add a pronunciation rule', page:'generate', awr:'registry', target:'#awr-add-btn', text:'The registry lists your rules and how often they appear in the manuscript. Click Add entry, or click Next and the tour will open the entry form for you.'},
            {title:'Original word or phrase', page:'generate', awr:'entry', target:'#awr-original-input', text:'Enter the exact word or phrase as it appears in your manuscript. Matching ignores capitalization but includes punctuation.'},
            {title:'Replacement pronunciation', page:'generate', awr:'entry', target:'#awr-replacement-input', text:'Enter how the engine should receive the word, such as a phonetic spelling. Matching is by substring, not whole word: choose a specific phrase so you do not accidentally change parts of other words.'},
            {title:'Choose a preview engine', page:'generate', awr:'entry', target:'#awr-preview-engine', text:'Choose an engine for testing the pronunciation. This selection is only for the preview; it does not change the narration engine or the speaker assignments in your job.'},
            {title:'Choose a preview voice', page:'generate', awr:'entry', target:'#awr-preview-voice', text:'Choose a compatible voice for the test. Pronunciations can vary by voice and engine, so test with the combination you intend to use. This does not change your cast.'},
            {title:'Compare the pronunciations', page:'generate', awr:'entry', target:'#awr-play-original', text:'Use Play original and Play replacement to hear each version. Select a voice and enter both words first. Preview synthesizes audio, so cloud previews may be billable.'},
            {title:'Save or cancel the rule', page:'generate', awr:'entry', target:'#awr-entry-ok', text:'Click OK to add a rule, or Cancel to leave it out. You can also click Next without saving; the tour never saves an entry automatically.'},
            {title:'Review your registry', page:'generate', awr:'registry', target:'#awr-table', text:'The In text column counts matches in the current manuscript. Rules apply in order, and a later rule can also match an earlier replacement. Edit or remove an entry as needed; then click Next.'},
            {title:'Save your progress', page:'generate', awr:'leave', target:'#project-manage-btn', text:'Use Projects to save the manuscript, cast, and registry. A project is not a replacement for an external backup. Test a short passage before starting a full book.'}
        ],
        review: [
            {title:'Find your production', page:'library', reviewAction:'find', target:'#library-search-input', fallback:'#library-list', refreshLibrary:true, text:'Search for a finished production by title. The tour uses the first visible production, or an already-open review. Choose a short production to keep the tour responsive.'},
            {title:'Open a production', page:'library', reviewAction:'expand', target:'.library-item.is-open .library-item-toggle', fallback:'#library-list', text:'The production is expanded to show its chapters and Full Story. Use the play button to listen, or the arrow to collapse it.'},
            {title:'Choose what to review', page:'library', reviewAction:'expand', target:'.library-item.is-open .library-chapter-review', fallback:'#library-list', text:'Each chapter has Review chunks. Full Story reviews the whole production. Click one yourself, or click Next to open the first available section.'},
            {title:'Review the speakers', page:'library', reviewAction:'open', target:'.bulk-speaker-summary', fallback:'#chunk-review-modal', text:'The review panel lists speakers and their chunk counts. It may take a moment to load. Expand a speaker to see voice and effects controls.'},
            {title:'Speaker properties', page:'library', reviewAction:'speaker', target:'.bulk-speaker-card.is-open .library-speaker-profile', fallback:'.bulk-speaker-summary', text:'Character profile, voice type and design prompt are saved for this production. Save changes before regenerating; saving alone does not change audio.'},
            {title:'Change a speaker’s voice', page:'library', reviewAction:'speaker', target:'.bulk-speaker-card.is-open .bulk-regen-section', fallback:'.bulk-speaker-summary', text:'Choose an engine and voice, preview the voice when available, and select the speaker to enable Regenerate all. This replaces that speaker’s takes in the reviewed scope and cloud synthesis can cost money. The tour will not start it.'},
            {title:'Find a passage', page:'library', reviewAction:'chunk', target:'.library-chunk-card.is-open .library-chunk-summary', fallback:'#chunk-review-modal', text:'The tour opens the first available chapter and chunk. Use play to audition a passage; Edit opens its text and generation controls.'},
            {title:'Edit text and direction', page:'library', reviewAction:'chunk', target:'.library-chunk-card.is-open .chunk-text-col', fallback:'#chunk-review-modal', text:'Correct the spoken text or, for supported engines, its voice direction. Choose the engine and voice under Regenerate with. These edits do not create new audio until you regenerate.'},
            {title:'Repair a chunk', page:'library', reviewAction:'chunk', target:'.library-chunk-card.is-open .library-chunk-regen', fallback:'#chunk-review-modal', text:'Regenerate creates a replacement take for this chunk using the chosen settings. Listen to the result before rebuilding. The tour will not regenerate or incur charges.'},
            {title:'Effects and silence', page:'library', reviewAction:'chunk', target:'.library-chunk-card.is-open .chunk-fx-section', fallback:'#chunk-review-modal', text:'Use speed, pitch and silence controls for adjustments. Preview and Apply FX are separate actions; the tour leaves your audio unchanged.'},
            {title:'Recompile the audio', page:'library', reviewAction:'open', target:'#chunk-review-recompile-btn', fallback:'#chunk-review-modal', text:'After repairing chunks, use Recompile audio to merge the current chunks into the output. A chapter review can also offer Rebuild chapter. Rebuilding merges existing takes; it does not synthesize them.'},
            {title:'Download and metadata', page:'library', reviewAction:'export', target:'.library-item.is-open .library-download-btn', fallback:'#library-list', text:'Use Download for available exports and Edit metadata for the production details. Listen to the rebuilt output before delivering it. The tour does not download or modify anything.'}
        ]
    };
    let tourId = 'first', steps = tours.first, index = 0, active = false, target = null;
    let card, highlight, pointer, previousAwr = null, tourHiddenEntry = false;
    let reviewProduction = null, reviewRequested = false;

    function prepareReviewStep(step) {
        if (!step.reviewAction) return;
        const panel = document.getElementById('chunk-review-modal');
        const openPanel = panel && !panel.classList.contains('hidden') && panel.closest('.library-item');
        if (openPanel) reviewProduction = openPanel;
        if (!reviewProduction || !reviewProduction.isConnected) reviewProduction = document.querySelector('#library-items .library-item');
        if (!reviewProduction || step.reviewAction === 'find') return;
        const toggle = reviewProduction.querySelector('.library-item-toggle');
        if (toggle && toggle.getAttribute('aria-expanded') !== 'true') toggle.click();
        if (step.reviewAction === 'expand' || step.reviewAction === 'export') return;
        if (!openPanel && !reviewRequested) {
            const button = reviewProduction.querySelector('.library-chapter-review');
            if (button) { reviewRequested = true; button.click(); }
        }
        if (!panel || panel.classList.contains('hidden')) return;
        if (step.reviewAction === 'speaker') {
            const speaker = panel.querySelector('.bulk-speaker-summary');
            if (speaker && speaker.getAttribute('aria-expanded') !== 'true') speaker.querySelector('.bulk-expand-toggle')?.click();
        }
        if (step.reviewAction === 'chunk') {
            const book = panel.querySelector('.book-header');
            if (book && book.getAttribute('aria-expanded') !== 'true') book.click();
            const chapter = panel.querySelector('.chapter-header');
            if (chapter && chapter.getAttribute('aria-expanded') !== 'true') chapter.click();
            const edit = panel.querySelector('.library-chunk-edit');
            if (edit && edit.getAttribute('aria-expanded') !== 'true') edit.click();
        }
    }
    const safeStore = (value) => { try { value === null ? localStorage.removeItem(storageKey) : localStorage.setItem(storageKey, JSON.stringify({tour: tourId, step: value})); } catch (_) {} };
    const readStore = () => { try { return localStorage.getItem(storageKey); } catch (_) { return null; } };

    function resolveTarget(step) {
        const preferred = document.querySelector(step.target);
        if (preferred) {
            const rect = preferred.getBoundingClientRect();
            if (rect.width && rect.height) return preferred;
        }
        return (step.fallback && document.querySelector(step.fallback)) || preferred;
    }

    function position() {
        if (!active || !target) return;
        const r = target.getBoundingClientRect();
        const visible = r.width && r.height && r.bottom > 0 && r.top < innerHeight;
        highlight.hidden = pointer.hidden = !visible;
        if (!visible) return;
        Object.assign(highlight.style, { left: `${Math.max(0, r.left - 4)}px`, top: `${Math.max(0, r.top - 4)}px`, width: `${Math.min(innerWidth, r.width + 8)}px`, height: `${Math.min(innerHeight, r.height + 8)}px` });
        Object.assign(pointer.style, { left: `${Math.max(2, Math.min(innerWidth - 35, r.left - 25))}px`, top: `${Math.max(0, Math.min(innerHeight - 35, r.top - 22))}px` });
        card.style.right = r.left > innerWidth / 2 ? 'auto' : '24px';
        card.style.left = r.left > innerWidth / 2 ? '24px' : 'auto';
    }
    function updateTarget() {
        const step = steps[index];
        const oldTarget = target;
        prepareReviewStep(step);
        target = resolveTarget(step);
        if (step.reviewAction) {
            const review = document.getElementById('chunk-review-modal');
            if (step.reviewAction !== 'find' && !reviewProduction) card.querySelector('p').textContent = 'No production is visible. Finish a job or clear the Library search, then return to this step.';
            else if (reviewProduction && !['find', 'expand', 'export'].includes(step.reviewAction) && !reviewProduction.querySelector('.library-chapter-review')) card.querySelector('p').textContent = 'This production has no chunk review available. Search for a production with saved chunks and restart this tour.';
            else if (review?.querySelector('.chunk-review-error, .chunk-review-empty')) card.querySelector('p').textContent = 'The review could not load chunks for this production. Choose another production with saved chunks and restart the tour.';
            else card.querySelector('p').textContent = step.text;
            if (target && target !== oldTarget) target.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        }
        if (step.library) {
            const found = !!document.querySelector(step.target);
            card.querySelector('p').textContent = found ? step.text : 'Your finished audio will appear here. If nothing is visible, wait for the job to finish, refresh the Library, and clear any search filter.';
        }
        position();
    }
    function show() {
        const step = steps[index];
        if (step.reviewAction && document.getElementById('chunk-review-modal')?.classList.contains('hidden')) reviewRequested = false;
        if (step.page) {
            const tab = document.querySelector(`.tab-button[data-tab="${step.page}"]`);
            if (tab && !tab.classList.contains('active')) tab.click();
        }
        if (step.section) document.querySelector(`[data-settings-section="${step.section}"]`)?.click();
        if (step.engine) document.querySelector(`[data-engine-tab="${step.engine}"]`)?.click();
        const registry = document.getElementById('alt-word-registry-modal');
        const entry = document.getElementById('awr-entry-modal');
        const isOpen = element => element && !element.classList.contains('hidden');
        if (step.awr === 'registry' || step.awr === 'entry') {
            if (!isOpen(registry) && typeof openAltWordRegistryModal === 'function') openAltWordRegistryModal();
            if (step.awr === 'registry' && isOpen(entry) && typeof closeAwrEntryModal === 'function') {
                // Hide rather than save or discard an unfinished rule. Back restores it.
                closeAwrEntryModal();
                tourHiddenEntry = true;
            }
            if (step.awr === 'entry' && previousAwr !== 'entry' && !isOpen(entry)) {
                if (tourHiddenEntry) {
                    document.getElementById('awr-entry-overlay')?.classList.remove('hidden');
                    entry?.classList.remove('hidden');
                    tourHiddenEntry = false;
                } else if (typeof openAwrEntryModal === 'function') openAwrEntryModal(-1);
            }
        } else if (step.awr === 'leave' && isOpen(registry) && !isOpen(entry) && typeof closeAltWordRegistryModal === 'function') {
            closeAltWordRegistryModal();
        }
        previousAwr = step.awr || null;
        if (step.subtab) {
            const subtab = document.querySelector(`[data-voices-target="${step.subtab}"]`);
            if (subtab && subtab.getAttribute('aria-selected') !== 'true') subtab.click();
        }
        target = resolveTarget(step);
        card.querySelector('.tutorial-count').textContent = `Step ${index + 1} of ${steps.length}`;
        card.querySelector('h2').textContent = step.title;
        card.querySelector('p').textContent = step.text;
        prepareReviewStep(step);
        target = resolveTarget(step);
        card.querySelector('.tutorial-feedback').textContent = '';
        card.querySelector('[data-tutorial-sample]').hidden = !step.sample;
        card.querySelector('[data-tutorial-examples]').hidden = !step.examples;
        card.querySelector('[data-tutorial-back]').disabled = index === 0;
        card.querySelector('[data-tutorial-next]').textContent = index === steps.length - 1 ? 'Finish' : 'Next';
        safeStore(index);
        target?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        requestAnimationFrame(updateTarget);
        if ((step.library || step.refreshLibrary) && typeof loadLibrary === 'function') loadLibrary();
    }
    function close() {
        active = false; target = null; safeStore(null);
        card.hidden = highlight.hidden = pointer.hidden = true;
    }
    function start(startIndex = 0, selectedTour = 'first') {
        previousAwr = null; tourHiddenEntry = false;
        reviewProduction = null; reviewRequested = false;
        tourId = Object.hasOwn(tours, selectedTour) ? selectedTour : 'first';
        steps = tours[tourId];
        index = Math.max(0, Math.min(steps.length - 1, startIndex));
        active = true; card.hidden = false; show();
        card.querySelector('h2').focus();
    }
    function initialize() {
        highlight = document.createElement('div'); highlight.className = 'tutorial-highlight'; highlight.hidden = true;
        pointer = document.createElement('div'); pointer.className = 'tutorial-pointer'; pointer.textContent = '➜'; pointer.hidden = true; pointer.setAttribute('aria-hidden', 'true');
        card = document.createElement('section'); card.className = 'tutorial-card'; card.hidden = true;
        card.setAttribute('role', 'region'); card.setAttribute('aria-label', 'Guided tutorial');
        card.innerHTML = '<div class="tutorial-count"></div><h2 tabindex="-1"></h2><p></p><div class="tutorial-examples" data-tutorial-examples hidden><div><button type="button" class="btn btn-secondary" data-example="raw">Insert raw example</button> <a href="/static/tutorial-examples/lighthouse-raw.txt" download>Download raw TXT</a></div><div><button type="button" class="btn btn-secondary" data-example="tagged">Insert tagged example</button> <a href="/static/tutorial-examples/lighthouse-tagged.txt" download>Download tagged TXT</a></div></div><p class="tutorial-example-status" role="status" aria-live="polite"></p><p class="tutorial-feedback" role="status" aria-live="polite"></p><div class="tutorial-actions"><button type="button" class="btn btn-secondary" data-tutorial-sample hidden>Use sample text</button><button type="button" class="btn btn-secondary" data-tutorial-close>Exit tour</button><button type="button" class="btn btn-secondary" data-tutorial-back>Back</button><button type="button" class="btn btn-primary" data-tutorial-next>Next</button></div>';
        document.body.append(highlight, pointer, card);
        card.querySelector('[data-tutorial-close]').addEventListener('click', close);
        card.querySelector('[data-tutorial-sample]').addEventListener('click', () => {
            const input = document.getElementById('input-text');
            if (!input || input.value === sampleStory) return;
            if (input.value.trim() && !window.confirm('Replace the current manuscript with the tutorial sample? Your existing text will be removed from the editor.')) return;
            input.value = sampleStory;
            input.dispatchEvent(new Event('input', { bubbles: true }));
        });
        for (const kind of ['raw', 'tagged']) {
            card.querySelector(`[data-example="${kind}"]`).addEventListener('click', async () => {
                const input = document.getElementById('input-text');
                if (!input) return;
                const previous = input.value;
                if (previous.trim() && !window.confirm('Replace the current manuscript with the tutorial example? Your existing text will be removed from the editor.')) return;
                const status = card.querySelector('.tutorial-example-status');
                status.textContent = 'Loading example…';
                try {
                    const response = await fetch(`/static/tutorial-examples/lighthouse-${kind}.txt`);
                    if (!response.ok) throw new Error('Example file could not be loaded.');
                    const text = await response.text();
                    if (!active || tourId !== 'manuscript' || input.value !== previous) {
                        status.textContent = 'The editor changed while loading; the example was not inserted.';
                        return;
                    }
                    input.value = text;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    status.textContent = `${kind === 'raw' ? 'Raw' : 'Tagged'} example inserted.`;
                } catch (_) { status.textContent = 'Could not load the example. Check the connection to TTS-Story and try again.'; }
            });
        }
        function move(delta) {
            const next = index + delta;
            if (next < 0) return;
            if (next >= steps.length) close(); else { index = next; show(); }
        }
        card.querySelector('[data-tutorial-back]').addEventListener('click', () => move(-1));
        card.querySelector('[data-tutorial-next]').addEventListener('click', () => move(1));
        document.getElementById('first-run-tutorial-btn')?.addEventListener('click', async () => { await dismissFirstRunWelcome(false); start(); });
        const tourDialog = document.getElementById('guided-tours-dialog');
        const openTours = () => { if (tourDialog && !tourDialog.open) tourDialog.showModal(); };
        document.getElementById('guided-tours-shortcut')?.addEventListener('click', openTours);
        document.getElementById('guided-tours-help-btn')?.addEventListener('click', openTours);
        document.getElementById('guided-tours-close-btn')?.addEventListener('click', () => tourDialog?.close());
        document.getElementById('tutorial-replay-btn')?.addEventListener('click', () => { tourDialog?.close(); start(); });
        for (const name of ['settings', 'voices', 'manuscript', 'review']) {
            document.getElementById(`tutorial-${name}-btn`)?.addEventListener('click', () => { tourDialog?.close(); start(0, name); });
        }
        window.addEventListener('resize', position); window.addEventListener('scroll', position, true);
        const libraryList = document.getElementById('library-items');
        // Watch card replacement, not every change inside a potentially huge chunk review.
        if (libraryList) new MutationObserver(() => { if (active && steps[index].page === 'library') updateTarget(); }).observe(libraryList, { childList: true });
        const reviewBody = document.getElementById('chunk-review-modal-body');
        if (reviewBody) new MutationObserver(() => { if (active && steps[index].reviewAction) requestAnimationFrame(updateTarget); }).observe(reviewBody, { childList: true });
        document.addEventListener('click', () => { if (active) requestAnimationFrame(updateTarget); });
        const saved = readStore();
        if (saved !== null) {
            try {
                const state = JSON.parse(saved);
                if (Number.isInteger(state) && state >= 0) start(state);
                else if (state && Object.hasOwn(tours, state.tour) && Number.isInteger(state.step) && state.step >= 0) start(state.step, state.tour);
            } catch (_) { /* Ignore stale or invalid progress. */ }
        }
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize); else initialize();
})();
