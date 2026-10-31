const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('the tutorial follows the queue and highlights an asynchronously loaded library item', () => {
    let activeTab = 'generate', item = null, observer, refreshes = 0;
    const clicks = [];
    function element() {
        const listeners = {};
        const children = {};
        return {
            style: {}, hidden: false, textContent: '', className: '',
            setAttribute() {}, focus() {}, scrollIntoView() {},
            getBoundingClientRect: () => ({left: 50, top: 50, bottom: 100, width: 100, height: 50}),
            addEventListener: (name, cb) => { listeners[name] = cb; },
            click: () => listeners.click?.(),
            querySelector: selector => children[selector] ||= element(),
            set innerHTML(value) {},
        };
    }
    const tabs = Object.fromEntries(['generate', 'queue', 'library'].map(name => [name, {
        ...element(),
        classList: {contains: () => activeTab === name},
        click: () => { activeTab = name; clicks.push(name); },
    }]));
    const fallback = element(), queue = element(), library = element();
    const shortcut = element(), helpButton = element(), closeButton = element(), startButton = element();
    const dialog = {open: false, showModal() { this.open = true; }, close() { this.open = false; }};
    const byId = {'library-items': library, 'guided-tours-dialog': dialog,
        'guided-tours-shortcut': shortcut, 'guided-tours-help-btn': helpButton,
        'guided-tours-close-btn': closeButton, 'tutorial-replay-btn': startButton};
    let card;
    const doc = {
        readyState: 'complete',
        addEventListener() {},
        createElement: element,
        body: {append: (...nodes) => { card = nodes[2]; }},
        getElementById: id => byId[id] || null,
        querySelector: selector => {
            const tab = selector.match(/^\.tab-button\[data-tab="(\w+)"\]$/);
            if (tab) return tabs[tab[1]];
            if (selector === '#queue-list') return queue;
            if (selector === '#library-list') return fallback;
            if (selector === '#library-items .library-item:first-child') return item;
            return null;
        },
    };
    const context = {
        document: doc, window: {addEventListener() {}},
        localStorage: {getItem: () => '9', setItem() {}, removeItem() {}},
        MutationObserver: class { constructor(cb) { observer = cb; } observe() {} },
        requestAnimationFrame: fn => fn(), innerWidth: 1200, innerHeight: 800,
        loadLibrary: () => { refreshes++; },
    };
    vm.runInNewContext(fs.readFileSync('static/js/tutorial.js', 'utf8'), context);
    activeTab = 'queue'; // Generate Audio navigated here before Next was clicked.
    card.querySelector('[data-tutorial-next]').click();
    assert.equal(activeTab, 'queue');
    assert.deepEqual(clicks, []);
    assert.equal(card.querySelector('h2').textContent, 'Follow your job');
    card.querySelector('[data-tutorial-next]').click();
    assert.equal(card.querySelector('h2').textContent, 'Open the Library');
    tabs.library.click();
    card.querySelector('[data-tutorial-next]').click();
    assert.equal(refreshes, 1);
    assert.match(card.querySelector('p').textContent, /nothing is visible/i);
    item = element();
    observer();
    assert.match(card.querySelector('p').textContent, /first visible production/i);
    assert.equal(card.querySelector('[data-tutorial-next]').textContent, 'Finish');
    shortcut.click();
    assert.equal(dialog.open, true);
    closeButton.click();
    assert.equal(dialog.open, false);
    helpButton.click();
    assert.equal(dialog.open, true);
    startButton.click();
    assert.equal(dialog.open, false);
    assert.equal(card.querySelector('h2').textContent, 'Choose your setup');
});

test('manuscript tour permits optional entry, preserves drafts on Back, and does not reopen after Cancel', () => {
    const buttons = {}, elements = {};
    function element() {
        const listeners = {}, children = {};
        return {style: {}, textContent: '', hidden: false, disabled: false,
            classList: {contains: name => name === 'hidden' && !!elements.hidden},
            setAttribute() {}, focus() {}, scrollIntoView() {},
            getBoundingClientRect: () => ({left: 10, top: 10, bottom: 60, width: 100, height: 50}),
            addEventListener: (name, cb) => { listeners[name] = cb; },
            click: () => listeners.click?.(), querySelector: key => children[key] ||= element(),
            set innerHTML(value) {}};
    }
    const registry = element(), entry = element();
    let registryOpen = false, entryOpen = false, entryOpens = 0, card;
    registry.classList = {contains: () => !registryOpen};
    entry.classList = {contains: () => !entryOpen, remove: () => {entryOpen = true;}};
    const tab = element(); tab.classList = {contains: () => true};
    const doc = {readyState:'complete', createElement: element, addEventListener() {},
        body: {append: (...nodes) => { card = nodes[2]; }},
        getElementById: id => ({'alt-word-registry-modal':registry, 'awr-entry-modal':entry})[id] || null,
        querySelector: selector => selector.startsWith('.tab-button') ? tab : (buttons[selector] ||= element())};
    const context = {document:doc, window:{addEventListener() {}}, localStorage:{getItem:()=>'{"tour":"manuscript","step":8}', setItem() {}, removeItem() {}},
        requestAnimationFrame: fn => fn(), innerWidth:1200, innerHeight:800,
        openAltWordRegistryModal: () => {registryOpen = true;},
        closeAltWordRegistryModal: () => {registryOpen = false;},
        openAwrEntryModal: () => {entryOpen = true; entryOpens++;},
        closeAwrEntryModal: () => {entryOpen = false;}};
    vm.runInNewContext(fs.readFileSync('static/js/tutorial.js', 'utf8'), context);
    const next = card.querySelector('[data-tutorial-next]');
    next.click();
    assert.equal(registryOpen, true);
    assert.equal(card.querySelector('h2').textContent, 'Add a pronunciation rule');
    next.click();
    assert.equal(entryOpen, true);
    assert.equal(entryOpens, 1);
    card.querySelector('[data-tutorial-back]').click();
    assert.equal(entryOpen, false);
    next.click();
    assert.equal(entryOpen, true);
    assert.equal(entryOpens, 1); // Restored, not reset by opening a new entry.
    for (let i = 0; i < 5; i++) next.click();
    assert.equal(card.querySelector('h2').textContent, 'Save or cancel the rule');
    assert.equal(entryOpens, 1);
    next.click();
    assert.equal(card.querySelector('h2').textContent, 'Review your registry');
    assert.equal(entryOpen, false);
    card.querySelector('[data-tutorial-back]').click();
    assert.equal(entryOpen, true);
    entryOpen = false; // User cancels; subsequent entry steps must not reopen it.
    card.querySelector('[data-tutorial-back]').click();
    assert.equal(entryOpen, false);
    next.click();
    assert.equal(entryOpen, false);
    next.click();
    next.click();
    assert.equal(registryOpen, false);
    assert.equal(card.querySelector('h2').textContent, 'Save your progress');
});

test('review tour expands the production and opens review controls after asynchronous loading', () => {
    let card, reviewObserver, expanded = false, panelOpen = false, loaded = false, speakerOpen = false, chapterOpen = false, chunkOpen = false, reviewClicks = 0;
    function element() {
        const listeners = {}, children = {};
        return {style:{}, hidden:false, textContent:'', isConnected:true,
            classList:{contains: () => false}, setAttribute(){}, focus(){}, scrollIntoView(){},
            getBoundingClientRect: () => ({left:10, top:10, bottom:60, width:100, height:50}),
            addEventListener:(name, cb) => {listeners[name]=cb;}, click:() => listeners.click?.(),
            querySelector:key => children[key] ||= element(), set innerHTML(value){}};
    }
    const toggle = element(); toggle.getAttribute = () => String(expanded); toggle.click = () => {expanded = !expanded;};
    const reviewButton = element(); reviewButton.click = () => {reviewClicks++; panelOpen = true;};
    const production = element(); production.querySelector = selector => selector === '.library-item-toggle' ? toggle : selector === '.library-chapter-review' ? reviewButton : null;
    const speakerToggle = element(); speakerToggle.click = () => {speakerOpen = true;};
    const speaker = element(); speaker.getAttribute = () => String(speakerOpen); speaker.querySelector = () => speakerToggle;
    const chapter = element(); chapter.getAttribute = () => String(chapterOpen); chapter.click = () => {chapterOpen = true;};
    const edit = element(); edit.getAttribute = () => String(chunkOpen); edit.click = () => {chunkOpen = true;};
    const panel = element(); panel.classList = {contains: () => !panelOpen}; panel.closest = () => panelOpen ? production : null;
    panel.querySelector = selector => !loaded ? null : ({'.bulk-speaker-summary':speaker, '.chapter-header':chapter, '.library-chunk-edit':edit})[selector] || null;
    const body = element(), tab = element(); tab.classList = {contains: () => true};
    const doc = {readyState:'complete', createElement:element, addEventListener(){}, body:{append:(...nodes) => {card=nodes[2];}},
        getElementById: id => ({'chunk-review-modal':panel, 'chunk-review-modal-body':body})[id] || null,
        querySelector: selector => selector.startsWith('.tab-button') ? tab : selector === '#library-items .library-item' ? production : null};
    const context = {document:doc, window:{addEventListener(){}}, localStorage:{getItem:()=>'{"tour":"review","step":0}', setItem(){}, removeItem(){}},
        MutationObserver: class { constructor(cb) {reviewObserver=cb;} observe(){} }, requestAnimationFrame:fn=>fn(), innerWidth:1200, innerHeight:800};
    vm.runInNewContext(fs.readFileSync('static/js/tutorial.js','utf8'), context);
    const next = card.querySelector('[data-tutorial-next]');
    next.click(); assert.equal(expanded, true);
    next.click(); next.click(); assert.equal(reviewClicks, 1);
    next.click(); assert.equal(speakerOpen, false);
    loaded = true; reviewObserver(); assert.equal(speakerOpen, true);
    next.click(); next.click();
    assert.equal(chapterOpen, true); assert.equal(chunkOpen, true);
    assert.equal(reviewClicks, 1);
});
