(function () {
  // Version switcher. Reads this page's own version from
  // window.FLASHDRR_DOCS (emitted by docs/conf.py via a layout template
  // hook) and queries the GitHub Releases API to build the dropdown.
  //
  // window.FLASHDRR_DOCS = { version: "0.5.1", is_latest: false }
  // window.FLASHDRR_REPO  = "pcarnah/flashdrr"   (optional override)
  //
  // Failure modes are all silent: the dropdown simply does not appear.

  var ctx = window.FLASHDRR_DOCS || {};
  var repo = window.FLASHDRR_REPO || (location.hostname.endsWith('github.io')
    ? location.pathname.split('/').filter(Boolean)[0] + '/'
        + (location.pathname.split('/').filter(Boolean)[1] || '')
    : '');
  // Fall back to the canonical repo if we can't infer it from the hostname.
  if (!repo || repo.split('/').length !== 2) repo = 'pcarnah/flashdrr';

  function sitePrefix() {
    var p = location.pathname;
    if (p.indexOf('/versions/') === 0) return '../../';
    if (p.indexOf('/latest/') === 0 || p.indexOf('/stable/') === 0) return '../';
    return '';
  }
  function pageVersion() {
    return (ctx && typeof ctx.version === 'string') ? ctx.version : '';
  }

  function semverCmp(a, b) {
    var pa = a.replace(/^v/, '').split(/[.+-]/);
    var pb = b.replace(/^v/, '').split(/[.+-]/);
    for (var i = 0; i < Math.max(pa.length, pb.length); i++) {
      var x = pa[i] || '', y = pb[i] || '';
      var xi = parseInt(x, 10), yi = parseInt(y, 10);
      var bothNum = !isNaN(xi) && !isNaN(yi) && /^\d+$/.test(x) && /^\d+$/.test(y);
      if (bothNum) { if (xi !== yi) return xi - yi; continue; }
      if (x < y) return -1; if (x > y) return 1;
    }
    return 0;
  }

  function buildDropdown(releases) {
    var prefix = sitePrefix();
    var current = pageVersion();
    var isLatest = !!(ctx && ctx.is_latest);
    // For the moving main build the dropdown's "current" entry is the
    // "latest" alias, not the literal version string.
    var here = isLatest ? 'latest' : current;
    // Normalize: releases API returns tag_name like "0.5.1" or "v0.5.1".
    var versions = releases.map(function (r) {
      return {
        version: r.tag_name.replace(/^v/, ''),
        isPrerelease: !!r.prerelease,
        ref: r.tag_name,
      };
    });
    // Sort newest-first using semver-ish comparison, then alphabetic fallback.
    versions.sort(function (a, b) {
      var c = semverCmp(b.version, a.version);
      if (c !== 0) return c;
      return b.version.localeCompare(a.version);
    });

    // Stable = newest non-prerelease; fall back to newest overall.
    var stable = null;
    for (var i = 0; i < versions.length; i++) {
      if (!versions[i].isPrerelease) { stable = versions[i]; break; }
    }
    if (!stable && versions.length) stable = versions[0];

    function hrefFor(value) {
      if (value === 'stable') return prefix + 'stable/';
      if (value === 'latest') return prefix + 'latest/';
      return prefix + 'versions/' + value.replace(/^v/, '') + '/';
    }

    var wrap = document.createElement('div');
    wrap.id = 'flashdrr-versions';
    wrap.style.cssText = 'background:#f8f9fa;border-bottom:1px solid #e1e4e5;padding:.4rem 1rem;font-size:.85rem;text-align:right;';

    var label = document.createElement('label');
    label.textContent = 'Version: ';
    label.setAttribute('for', 'flashdrr-version-select');
    var sel = document.createElement('select');
    sel.id = 'flashdrr-version-select';

    function add(value, text) {
      var o = document.createElement('option');
      o.value = value;
      o.textContent = text;
      if (value === here || value === current) o.selected = true;
      sel.appendChild(o);
    }

    if (ctx.is_latest) add('latest', 'latest (development)');
    if (stable) add('stable', 'stable (' + stable.version + ')');
    versions.forEach(function (v) {
      // Skip if a release entry is already represented by stable or latest.
      if (stable && stable.version === v.version) return;
      add(v.ref, v.version + (v.isPrerelease ? ' (pre)' : ''));
    });

    sel.addEventListener('change', function () {
      location.href = hrefFor(sel.value);
    });

    wrap.appendChild(label);
    wrap.appendChild(sel);
    var target = document.querySelector('[role="main"]') || document.body;
    target.parentNode.insertBefore(wrap, target);
  }

  fetch('https://api.github.com/repos/' + repo + '/releases?per_page=30', {
    headers: { 'Accept': 'application/vnd.github+json' },
    cache: 'default',
  })
    .then(function (r) { return r.ok ? r.json() : null; })
    .catch(function () { return null; })
    .then(function (data) { if (Array.isArray(data)) buildDropdown(data); });
})();
