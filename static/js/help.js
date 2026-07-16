(function () {
    'use strict';

    const CATALOG_URL = '/api/help/catalog';
    const ARTICLE_URL = id => `/api/help/articles/${encodeURIComponent(id)}`;
    const HELP_HASH_PREFIX = '#help/';
    const MAX_SEARCH_RESULTS = 12;
    const APP_DOCUMENT_TITLE = document.title || 'TTS-Story';

    let catalog = null;
    let articlesById = new Map();
    let categoryByArticleId = new Map();
    let articleCache = new Map();
    let initialized = false;
    let initializationPromise = null;
    let eventsBound = false;
    let articleRequestSequence = 0;
    let currentArticleId = '';
    let previousTab = 'generate';

    function element(id) {
        return document.getElementById(id);
    }

    function resolveArticleId(rawId) {
        const id = String(rawId || '').trim().toLowerCase();
        if (!id || !catalog) return '';
        const resolved = catalog.aliases?.[id] || id;
        return articlesById.has(resolved) ? resolved : '';
    }

    function activeTabName() {
        return document.querySelector('.tab-button.active')?.dataset?.tab || 'generate';
    }

    function tabLabel(tabName) {
        return document.querySelector(`.tab-button[data-tab="${tabName}"]`)?.textContent?.trim() || 'application';
    }

    function switchToTab(tabName) {
        const button = document.querySelector(`.tab-button[data-tab="${tabName}"]`);
        if (!button) return false;
        button.click();
        return true;
    }

    function prefersReducedMotion() {
        return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
    }

    function scrollToElement(target, block = 'start') {
        target?.scrollIntoView({
            behavior: prefersReducedMotion() ? 'auto' : 'smooth',
            block,
        });
    }

    function focusElement(target, { scroll = false } = {}) {
        if (!target) return;
        if (!target.matches('button, a, input, select, textarea, [tabindex]')) {
            target.setAttribute('tabindex', '-1');
        }
        target.focus({ preventScroll: !scroll });
    }

    function setStatus(message, kind = 'loading', retry = null) {
        const status = element('help-center-status');
        if (!status) return;
        status.replaceChildren();
        if (message) {
            const copy = document.createElement('span');
            copy.textContent = message;
            status.appendChild(copy);
            if (typeof retry === 'function') {
                const retryButton = document.createElement('button');
                retryButton.type = 'button';
                retryButton.className = 'help-status-retry';
                retryButton.textContent = 'Retry';
                retryButton.addEventListener('click', retry);
                status.appendChild(retryButton);
            }
        }
        status.dataset.kind = kind;
        status.classList.toggle('hidden', !message);
    }

    function clearSearch() {
        const input = element('help-center-search');
        const results = element('help-center-search-results');
        const clearButton = element('help-center-search-clear');
        if (input) input.value = '';
        if (input) input.setAttribute('aria-expanded', 'false');
        if (results) {
            results.innerHTML = '';
            results.classList.add('hidden');
        }
        if (clearButton) clearButton.classList.add('hidden');
    }

    function parseHelpHash() {
        if (!window.location.hash.startsWith(HELP_HASH_PREFIX)) return '';
        try {
            return decodeURIComponent(window.location.hash.slice(HELP_HASH_PREFIX.length));
        } catch (_error) {
            return '';
        }
    }

    function setHelpHash(articleId, mode = 'push') {
        const url = `${window.location.pathname}${window.location.search}${HELP_HASH_PREFIX}${encodeURIComponent(articleId)}`;
        if (mode === 'replace') {
            window.history.replaceState({ helpArticle: articleId }, '', url);
        } else if (window.location.hash !== `${HELP_HASH_PREFIX}${encodeURIComponent(articleId)}`) {
            window.history.pushState({ helpArticle: articleId }, '', url);
        }
    }

    function clearHelpHash() {
        if (!window.location.hash.startsWith(HELP_HASH_PREFIX)) return;
        window.history.replaceState({}, '', `${window.location.pathname}${window.location.search}`);
    }

    function orderedArticleIds() {
        if (!catalog) return [];
        return catalog.categories.flatMap(category => category.article_ids || []);
    }

    function articleMeta(articleId) {
        return articlesById.get(resolveArticleId(articleId)) || null;
    }

    function makeTopicButton(article, className = 'help-topic-card') {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = className;
        button.dataset.helpArticle = article.id;

        const title = document.createElement('span');
        title.className = 'help-topic-card-title';
        title.textContent = article.title;

        const summary = document.createElement('span');
        summary.className = 'help-topic-card-summary';
        summary.textContent = article.summary;

        button.append(title, summary);
        return button;
    }

    function renderHome() {
        const home = element('help-center-home');
        const reader = element('help-center-reader');
        const categoriesHost = element('help-center-categories');
        if (!home || !reader || !categoriesHost || !catalog) return;

        articleRequestSequence += 1;
        setStatus('');
        currentArticleId = '';
        home.classList.remove('hidden');
        reader.classList.add('hidden');
        categoriesHost.innerHTML = '';

        catalog.categories.forEach(category => {
            const section = document.createElement('section');
            section.className = 'help-category-section';
            section.dataset.helpCategory = category.id;

            const headingRow = document.createElement('div');
            headingRow.className = 'help-category-heading';
            const headingText = document.createElement('div');
            const heading = document.createElement('h3');
            heading.textContent = category.title;
            const description = document.createElement('p');
            description.textContent = category.description;
            headingText.append(heading, description);
            const count = document.createElement('span');
            count.className = 'help-category-count';
            count.textContent = `${category.article_ids.length} ${category.article_ids.length === 1 ? 'guide' : 'guides'}`;
            headingRow.append(headingText, count);

            const grid = document.createElement('div');
            grid.className = 'help-topic-grid';
            category.article_ids.forEach(id => {
                const article = articlesById.get(id);
                if (article) grid.appendChild(makeTopicButton(article));
            });
            section.append(headingRow, grid);
            categoriesHost.appendChild(section);
        });

        updateSidebarSelection('');
        if (activeTabName() === 'help') {
            document.title = 'Help — TTS-Story';
        }
    }

    function renderSidebar() {
        const host = element('help-center-sidebar-nav');
        if (!host || !catalog) return;
        host.innerHTML = '';

        catalog.categories.forEach(category => {
            const group = document.createElement('section');
            group.className = 'help-sidebar-group';
            const heading = document.createElement('h3');
            heading.textContent = category.title;
            group.appendChild(heading);

            category.article_ids.forEach(id => {
                const article = articlesById.get(id);
                if (!article) return;
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'help-sidebar-link';
                button.dataset.helpArticle = article.id;
                button.textContent = article.title;
                group.appendChild(button);
            });
            host.appendChild(group);
        });
    }

    function updateSidebarSelection(articleId) {
        document.querySelectorAll('.help-sidebar-link').forEach(button => {
            const isActive = button.dataset.helpArticle === articleId;
            button.classList.toggle('active', isActive);
            if (isActive) {
                button.setAttribute('aria-current', 'page');
            } else {
                button.removeAttribute('aria-current');
            }
        });
        document.querySelectorAll('.help-sidebar-group').forEach(group => {
            group.classList.toggle('has-active', Boolean(group.querySelector('.help-sidebar-link.active')));
        });
    }

    function renderOnThisPage(body) {
        const host = element('help-center-toc');
        if (!host) return;
        host.innerHTML = '';
        const headings = Array.from(body.querySelectorAll('h2, h3'));
        host.classList.toggle('hidden', headings.length < 2);
        if (headings.length < 2) return;

        const title = document.createElement('strong');
        title.textContent = 'On this page';
        const list = document.createElement('ul');
        headings.forEach((heading, index) => {
            if (!heading.id) heading.id = `section-${index + 1}`;
            const item = document.createElement('li');
            item.className = heading.tagName === 'H3' ? 'help-toc-subitem' : '';
            const link = document.createElement('a');
            link.href = `#${heading.id}`;
            link.textContent = heading.textContent;
            link.addEventListener('click', event => {
                event.preventDefault();
                scrollToElement(heading);
                focusElement(heading);
            });
            item.appendChild(link);
            list.appendChild(item);
        });
        host.append(title, list);
    }

    function prepareArticleLinks(body) {
        body.querySelectorAll('a[href]').forEach(link => {
            const href = link.getAttribute('href') || '';
            if (href.startsWith('help:')) {
                link.dataset.helpArticle = href.slice(5);
                link.href = `${HELP_HASH_PREFIX}${encodeURIComponent(href.slice(5))}`;
                return;
            }
            if (href.startsWith('app:')) {
                link.dataset.helpAppLocation = href.slice(4);
                link.href = '#';
                return;
            }
            if (/^https?:\/\//i.test(href)) {
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                return;
            }
            if (!href.startsWith('#')) link.removeAttribute('href');
        });
    }

    function prepareArticleImages(body) {
        body.querySelectorAll('img').forEach(image => {
            const wrapper = image.parentElement;
            if (!wrapper || wrapper.tagName !== 'P' || wrapper.children.length !== 1) return;
            wrapper.classList.add('help-article-image');
            const caption = wrapper.nextElementSibling;
            if (
                caption?.tagName === 'P'
                && caption.children.length === 1
                && caption.firstElementChild?.tagName === 'EM'
            ) {
                caption.classList.add('help-article-image-caption');
            }
        });
    }

    function renderRelated(article) {
        const section = element('help-center-related');
        const host = element('help-center-related-links');
        if (!section || !host) return;
        host.innerHTML = '';
        const related = (article.related || []).map(articleMeta).filter(Boolean);
        section.classList.toggle('hidden', related.length === 0);
        related.forEach(entry => host.appendChild(makeTopicButton(entry, 'help-related-card')));
    }

    function renderArticleNavigation(articleId) {
        const previousButton = element('help-center-previous');
        const nextButton = element('help-center-next');
        if (!previousButton || !nextButton) return;
        const ids = orderedArticleIds();
        const index = ids.indexOf(articleId);
        const previous = index > 0 ? articlesById.get(ids[index - 1]) : null;
        const next = index >= 0 && index < ids.length - 1 ? articlesById.get(ids[index + 1]) : null;

        previousButton.classList.toggle('hidden', !previous);
        nextButton.classList.toggle('hidden', !next);
        if (previous) {
            previousButton.dataset.helpArticle = previous.id;
            previousButton.querySelector('span:last-child').textContent = previous.title;
        }
        if (next) {
            nextButton.dataset.helpArticle = next.id;
            nextButton.querySelector('span:last-child').textContent = next.title;
        }
    }

    async function fetchArticle(articleId) {
        if (articleCache.has(articleId)) return articleCache.get(articleId);
        const response = await fetch(ARTICLE_URL(articleId), { headers: { Accept: 'application/json' } });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload.success || !payload.article) {
            throw new Error(payload.error || `Unable to load help article (${response.status}).`);
        }
        articleCache.set(articleId, payload.article);
        return payload.article;
    }

    async function openArticle(rawId, options = {}) {
        const active = activeTabName();
        if (active !== 'help') previousTab = active;
        if (options.switchTab !== false) switchToTab('help');
        clearSearch();

        if (!catalog) {
            const ready = await initHelpSystem();
            if (!ready || !catalog) return false;
        }

        const articleId = resolveArticleId(rawId);
        if (!articleId) {
            clearHelpHash();
            renderHome();
            setStatus(`No help article is available for “${rawId}”. Choose a topic or search the guide.`, 'error');
            focusElement(element('help-center-status'));
            return false;
        }

        const requestId = ++articleRequestSequence;
        const requestedMeta = articlesById.get(articleId);

        const home = element('help-center-home');
        const reader = element('help-center-reader');
        if (home) home.classList.add('hidden');
        if (reader) reader.classList.remove('hidden');
        setStatus('Loading guide…');

        const breadcrumb = element('help-center-breadcrumb');
        const title = element('help-center-article-title');
        const summary = element('help-center-article-summary');
        const meta = element('help-center-article-meta');
        const body = element('help-center-article-body');
        const backButton = element('help-center-back-to-app');
        const category = categoryByArticleId.get(articleId);
        if (breadcrumb) breadcrumb.textContent = category?.title || 'User Guide';
        if (title) title.textContent = requestedMeta?.title || 'Loading guide…';
        if (summary) summary.textContent = requestedMeta?.summary || '';
        if (meta) meta.textContent = 'Loading…';
        if (body) body.replaceChildren();
        if (backButton) backButton.textContent = `Back to ${tabLabel(previousTab)}`;
        renderOnThisPage(body || document.createElement('div'));
        renderRelated({ related: [] });
        renderArticleNavigation('');
        updateSidebarSelection(articleId);

        try {
            const article = await fetchArticle(articleId);
            if (requestId !== articleRequestSequence) return false;
            currentArticleId = articleId;

            if (breadcrumb) breadcrumb.textContent = category?.title || 'User Guide';
            if (title) title.textContent = article.title;
            if (summary) summary.textContent = article.summary;
            if (meta) {
                meta.textContent = `${article.reading_minutes} min read · ${article.word_count.toLocaleString()} words`;
            }
            if (backButton) backButton.textContent = `Back to ${tabLabel(previousTab)}`;
            if (body) {
                body.innerHTML = article.html;
                prepareArticleLinks(body);
                prepareArticleImages(body);
                renderOnThisPage(body);
            }

            renderRelated(article);
            renderArticleNavigation(articleId);
            updateSidebarSelection(articleId);
            setStatus('');
            document.title = `${article.title} — TTS-Story Help`;

            if (options.updateHistory !== false) {
                setHelpHash(articleId, options.replaceHistory ? 'replace' : 'push');
            }
            if (options.focus !== false && title) {
                title.setAttribute('tabindex', '-1');
                title.focus({ preventScroll: false });
            } else if (options.scroll !== false) {
                element('help-center-reader')?.scrollIntoView({ block: 'start' });
            }
            return true;
        } catch (error) {
            if (requestId !== articleRequestSequence) return false;
            console.error('Unable to load help article', error);
            currentArticleId = '';
            if (meta) meta.textContent = '';
            if (summary) summary.textContent = 'This article could not be loaded. Retry the request or return to all topics.';
            if (body) body.replaceChildren();
            setStatus(
                error.message || 'Unable to load this guide.',
                'error',
                () => openArticle(articleId, { ...options, replaceHistory: true })
            );
            focusElement(element('help-center-status'));
            return false;
        }
    }

    function scoreSearch(article, tokens) {
        const title = article.title.toLowerCase();
        const summary = article.summary.toLowerCase();
        const keywords = (article.keywords || []).join(' ').toLowerCase();
        const body = (article.search_text || '').toLowerCase();
        let score = 0;
        for (const token of tokens) {
            if (!title.includes(token) && !summary.includes(token) && !keywords.includes(token) && !body.includes(token)) {
                return -1;
            }
            if (title.includes(token)) score += 12;
            if (keywords.includes(token)) score += 7;
            if (summary.includes(token)) score += 4;
            if (body.includes(token)) score += 1;
        }
        return score;
    }

    function renderSearchResults(query) {
        const host = element('help-center-search-results');
        const clearButton = element('help-center-search-clear');
        if (!host || !catalog) return;
        const normalized = String(query || '').trim().toLowerCase();
        if (clearButton) clearButton.classList.toggle('hidden', !normalized);
        host.innerHTML = '';
        if (!normalized) {
            host.classList.add('hidden');
            element('help-center-search')?.setAttribute('aria-expanded', 'false');
            return;
        }

        const tokens = normalized.split(/\s+/).filter(Boolean);
        const matches = catalog.articles
            .map(article => ({ article, score: scoreSearch(article, tokens) }))
            .filter(entry => entry.score >= 0)
            .sort((left, right) => right.score - left.score || left.article.title.localeCompare(right.article.title))
            .slice(0, MAX_SEARCH_RESULTS);

        if (!matches.length) {
            const empty = document.createElement('div');
            empty.className = 'help-search-empty';
            empty.textContent = 'No guide matched every search word. Try a provider, engine, page name, or error code.';
            host.appendChild(empty);
        } else {
            matches.forEach(({ article }) => {
                const category = categoryByArticleId.get(article.id);
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'help-search-result';
                button.dataset.helpArticle = article.id;

                const copy = document.createElement('span');
                const title = document.createElement('strong');
                title.textContent = article.title;
                const summary = document.createElement('small');
                summary.textContent = article.summary;
                copy.append(title, summary);

                const categoryLabel = document.createElement('span');
                categoryLabel.className = 'help-search-result-category';
                categoryLabel.textContent = category?.title || 'Guide';
                button.append(copy, categoryLabel);
                host.appendChild(button);
            });
        }
        host.classList.remove('hidden');
        element('help-center-search')?.setAttribute('aria-expanded', 'true');
    }

    function openAppLocation(rawLocation) {
        const [tabName, detail] = String(rawLocation || '').split('/');
        const tabButton = document.querySelector(`.tab-button[data-tab="${tabName || 'generate'}"]`);
        articleRequestSequence += 1;
        if (!switchToTab(tabName || 'generate')) return;
        clearHelpHash();

        if (tabName === 'settings' && detail) {
            const settingsGroup = document.getElementById('engine-settings-group');
            if (settingsGroup?.classList.contains('collapsed')) {
                settingsGroup.querySelector('[data-toggle="settings-group"]')?.click();
            }
            const engineButton = document.querySelector(`.engine-tab-btn[data-engine-tab="${detail}"]`);
            engineButton?.click();
            scrollToElement(document.getElementById(`engine-panel-${detail}`));
            focusElement(engineButton || tabButton);
        } else {
            focusElement(tabButton);
        }
    }

    function bindEvents() {
        document.addEventListener('click', event => {
            const contextual = event.target.closest('[data-help-id]');
            if (contextual) {
                event.preventDefault();
                event.stopPropagation();
                openArticle(contextual.dataset.helpId);
                return;
            }

            const articleLink = event.target.closest('[data-help-article]');
            if (articleLink) {
                event.preventDefault();
                openArticle(articleLink.dataset.helpArticle);
                return;
            }

            const appLink = event.target.closest('[data-help-app-location]');
            if (appLink) {
                event.preventDefault();
                openAppLocation(appLink.dataset.helpAppLocation);
            }
        });

        element('help-center-search')?.addEventListener('input', event => renderSearchResults(event.target.value));
        element('help-center-search')?.addEventListener('keydown', event => {
            if (event.key === 'Escape') {
                clearSearch();
                event.currentTarget.blur();
            }
        });
        element('help-center-search-clear')?.addEventListener('click', () => {
            clearSearch();
            element('help-center-search')?.focus();
        });
        element('help-center-home-button')?.addEventListener('click', () => {
            clearHelpHash();
            renderHome();
            scrollToElement(element('help-center-title'));
            focusElement(element('help-center-title'));
        });
        element('help-center-back-to-app')?.addEventListener('click', () => {
            articleRequestSequence += 1;
            clearHelpHash();
            const tabName = previousTab || 'generate';
            switchToTab(tabName);
            focusElement(document.querySelector(`.tab-button[data-tab="${tabName}"]`));
        });

        const helpTabButton = document.querySelector('.tab-button[data-tab="help"]');
        helpTabButton?.addEventListener('click', () => {
            const active = activeTabName();
            if (active !== 'help') previousTab = active;
        }, true);
        helpTabButton?.addEventListener('click', () => {
            if (!currentArticleId && catalog) renderHome();
        });
        document.querySelectorAll('.tab-button:not([data-tab="help"])').forEach(button => {
            button.addEventListener('click', () => {
                articleRequestSequence += 1;
                document.title = APP_DOCUMENT_TITLE;
                clearHelpHash();
            });
        });

        window.addEventListener('popstate', () => {
            const hashId = parseHelpHash();
            if (hashId) {
                openArticle(hashId, { updateHistory: false, focus: false });
            } else if (activeTabName() === 'help') {
                renderHome();
            }
        });

        document.addEventListener('keydown', event => {
            if (event.key === '/' && activeTabName() === 'help' && !/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) {
                event.preventDefault();
                element('help-center-search')?.focus();
            }
        });
    }

    async function loadCatalog() {
        setStatus('Loading the bundled user guide…');
        const response = await fetch(CATALOG_URL, { headers: { Accept: 'application/json' } });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || `Unable to load the user guide (${response.status}).`);
        }
        catalog = payload;
        articlesById = new Map(payload.articles.map(article => [article.id, article]));
        categoryByArticleId = new Map();
        payload.categories.forEach(category => {
            category.article_ids.forEach(id => categoryByArticleId.set(id, category));
        });
        const version = element('help-center-version');
        if (version) version.textContent = `Guide ${payload.version}`;
    }

    async function initialize() {
        if (initialized) return true;
        try {
            if (!eventsBound) {
                bindEvents();
                eventsBound = true;
            }
            await loadCatalog();
            renderSidebar();
            renderHome();
            setStatus('');
            const hashId = parseHelpHash();
            if (hashId) {
                await openArticle(hashId, {
                    updateHistory: false,
                    focus: false,
                    replaceHistory: true,
                });
            }
            initialized = true;
            return true;
        } catch (error) {
            initialized = false;
            catalog = null;
            articlesById = new Map();
            categoryByArticleId = new Map();
            console.error('Unable to initialize help center', error);
            setStatus(
                `${error.message || 'Unable to load the bundled guide.'} Retry, restart TTS-Story after updating, or read docs/help in the project folder.`,
                'error',
                () => initHelpSystem()
            );
            return false;
        }
    }

    function initHelpSystem() {
        if (!initializationPromise) {
            initializationPromise = initialize().then(success => {
                if (!success) initializationPromise = null;
                return success;
            });
        }
        return initializationPromise;
    }

    window.initHelpSystem = initHelpSystem;
    window.openHelpModal = openArticle;
    window.TTSStoryHelp = {
        init: initHelpSystem,
        open: openArticle,
        showHome: renderHome,
    };
})();
