// "Latest Blogs" — show the newest 3 posts from the blog's RSS feed
// (/swayam-script/index.xml). Same domain in production; works from localhost
// too because GitHub Pages serves the feed with Access-Control-Allow-Origin: *.
// If the feed can't be loaded, it degrades to a simple "visit blog" link.
(function () {
  var BLOG = "https://swayaminsync.github.io/swayam-script/";
  var FEED = BLOG + "index.xml";
  var container = document.getElementById("latest-posts");
  if (!container) return;

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function abs(url) { try { return new URL(url, BLOG).href; } catch (e) { return BLOG; } }
  function txt(parent, tag) {
    var e = parent.getElementsByTagName(tag)[0];
    return e ? e.textContent.trim() : "";
  }
  function fmtDate(s) {
    var d = s ? new Date(s) : null;
    return d && !isNaN(d)
      ? d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })
      : (s || "");
  }

  fetch(FEED, { credentials: "omit" })
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
    .then(function (xml) {
      var doc = new DOMParser().parseFromString(xml, "application/xml");
      if (doc.getElementsByTagName("parsererror").length) throw new Error("bad xml");
      var items = Array.prototype.slice.call(doc.getElementsByTagName("item"));
      if (!items.length) throw new Error("no items");

      var frag = document.createDocumentFragment();
      items.slice(0, 3).forEach(function (it) {
        var enc = it.getElementsByTagName("enclosure")[0] || it.getElementsByTagName("media:content")[0];
        var title = txt(it, "title");
        var link = txt(it, "link") || BLOG;
        var date = fmtDate(txt(it, "pubDate"));
        var cats = Array.prototype.map.call(it.getElementsByTagName("category"), function (c) {
          return c.textContent.trim();
        }).slice(0, 3);
        var img = enc ? abs(enc.getAttribute("url") || "") : "";

        var a = document.createElement("a");
        a.className = "blog-card";
        a.href = link; a.target = "_blank"; a.rel = "noopener";
        a.innerHTML =
          (img ? '<div class="blog-card__thumb"><img loading="lazy" src="' + esc(img) + '" alt=""></div>' : "") +
          '<div class="blog-card__body">' +
            '<div class="blog-card__title">' + esc(title) + "</div>" +
            '<div class="blog-card__meta">' + esc(date) +
              (cats.length ? " &middot; " + esc(cats.join(", ")) : "") +
            "</div>" +
          "</div>";
        frag.appendChild(a);
      });
      container.innerHTML = "";
      container.appendChild(frag);
    })
    .catch(function () {
      container.innerHTML =
        '<p class="blog-loading"><a href="' + BLOG + '" target="_blank" rel="noopener">' +
        "Read my latest posts on the blog &rarr;</a></p>";
    });
})();
