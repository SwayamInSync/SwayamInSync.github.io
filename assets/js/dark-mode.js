// Light/dark theme toggle.
// The initial theme is set inline in <head> (see _includes/head.html) to avoid
// a flash of the wrong theme on first paint. This file only handles the toggle
// button and reacting to OS-level changes when the user hasn't chosen explicitly.
(function () {
  var btn = document.getElementById('theme-toggle');
  if (!btn) return;

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('theme', theme); } catch (e) {}
  }

  btn.addEventListener('click', function () {
    applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  });

  // Follow the OS setting only while the user hasn't made an explicit choice.
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onChange = function (e) {
      try { if (localStorage.getItem('theme')) return; } catch (err) {}
      document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
    };
    if (mq.addEventListener) { mq.addEventListener('change', onChange); }
    else if (mq.addListener) { mq.addListener(onChange); }
  }
})();
