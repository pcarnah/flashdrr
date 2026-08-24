(function () {
  // The site layout is:
  //   site/
  //     versions.json
  //     latest/...
  //     stable/...
  //     versions/<slug>/...
  //
  // From any rendered page, versions.json lives at ../versions.json for the
  // alias pages and ../../versions.json for the per-tag pages. Compute the
  // relative prefix once and reuse it for both the fetch and the option hrefs.
  function sitePrefix() {
    var p = location.pathname;
    if (p.indexOf('/versions/') === 0) {
      // /versions/<slug>/<rest>  -> go up 2 levels
      return '../../';
    }
    if (p.indexOf('/latest/') === 0 || p.indexOf('/stable/') === 0) {
      return '../';
    }
    // Landing page or unknown.
    return '';
  }
  function hereSlug() {
    var p = location.pathname.replace(/\/$/, '');
    var m = p.match(/^\/(?:versions\/|)([^/]+)\/?$/);
    return m ? m[1] : '';
  }

  var prefix = sitePrefix();
  fetch(prefix + 'versions.json', { cache: 'no-store' })
    .then(function (r) { return r.ok ? r.json() : null; })
    .catch(function () { return null; })
    .then(function (data) {
      if (!data) return;
      var here = hereSlug();
      var versions = Object.keys(data.versions || {}).sort(function (a, b) {
        return (data.versions[b].version || '').localeCompare(
          data.versions[a].version || ''
        );
      });

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
        if (value === here) o.selected = true;
        sel.appendChild(o);
      }

      function hrefFor(value) {
        if (value === 'stable' || value === 'latest') {
          return prefix + value + '/';
        }
        return prefix + 'versions/' + value + '/';
      }

      if (data.stable) add('stable', 'stable (' + data.versions[data.stable].version + ')');
      if (data.latest) add('latest', 'latest (' + data.versions[data.latest].version + ')');
      versions.forEach(function (slug) {
        if (slug === data.stable || slug === data.latest) return;
        add(slug, data.versions[slug].version);
      });

      sel.addEventListener('change', function () {
        location.href = hrefFor(sel.value);
      });

      wrap.appendChild(label);
      wrap.appendChild(sel);
      var target = document.querySelector('[role="main"]') || document.body;
      target.parentNode.insertBefore(wrap, target);
    });
})();
