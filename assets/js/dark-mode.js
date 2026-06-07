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
})();
