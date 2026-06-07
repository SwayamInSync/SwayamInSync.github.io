// "Latest Blogs" — show the newest 3 posts from the Quarto blog.
// Primary source: the RSS feed (/swayam-script/index.xml). If that's
// unavailable, fall back to scraping the blog homepage. Both run client-side;
// same domain in production, and GitHub Pages sends Access-Control-Allow-Origin: *
// so it also works from localhost. Degrades to a plain "visit blog" link.
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

  // ---- parse the RSS feed ----
  function fromFeed(xmlText) {
    var doc = new DOMParser().parseFromString(xmlText, "application/xml");
    if (doc.getElementsByTagName("parsererror").length) throw new Error("bad xml");
    var items = Array.prototype.slice.call(doc.getElementsByTagName("item"));
    if (!items.length) throw new Error("no items");
    return items.slice(0, 3).map(function (it) {
      var enc = it.getElementsByTagName("enclosure")[0] || it.getElementsByTagName("media:content")[0];
      return {
        title: txt(it, "title"),
        link: txt(it, "link") || BLOG,
        date: fmtDate(txt(it, "pubDate")),
        cats: Array.prototype.map.call(it.getElementsByTagName("category"), function (c) {
          return c.textContent.trim();
        }).slice(0, 3),
        img: enc ? abs(enc.getAttribute("url") || "") : "",
      };
    });
  }

  // ---- fallback: scrape the blog homepage ----
  function fromHtml(html) {
    var doc = new DOMParser().parseFromString(html, "text/html");
    var posts = Array.prototype.slice.call(doc.querySelectorAll(".quarto-post"));
    if (!posts.length) throw new Error("no posts");
    posts.sort(function (a, b) {
      return (+b.getAttribute("data-listing-date-sort") || 0) -
             (+a.getAttribute("data-listing-date-sort") || 0);
    });
    return posts.slice(0, 3).map(function (p) {
      var t = p.querySelector(".listing-title a") || p.querySelector(".listing-title");
      var l = p.querySelector(".listing-title a[href]") || p.querySelector("a[href]");
      var img = p.querySelector(".thumbnail img, img.thumbnail-image");
      return {
        title: t ? t.textContent.trim() : "Untitled",
        link: l ? abs(l.getAttribute("href")) : BLOG,
        date: (p.querySelector(".listing-date") || {}).textContent ?
              p.querySelector(".listing-date").textContent.trim() : "",
        cats: Array.prototype.map.call(p.querySelectorAll(".listing-category"), function (c) {
          return c.textContent.trim();
        }).slice(0, 3),
        img: img ? abs(img.getAttribute("src")) : "",
      };
    });
  }

  function render(posts) {
    var frag = document.createDocumentFragment();
    posts.forEach(function (p) {
      var a = document.createElement("a");
      a.className = "blog-card";
      a.href = p.link; a.target = "_blank"; a.rel = "noopener";
      a.innerHTML =
        (p.img ? '<div class="blog-card__thumb"><img loading="lazy" src="' + esc(p.img) + '" alt=""></div>' : "") +
        '<div class="blog-card__body">' +
          '<div class="blog-card__title">' + esc(p.title) + "</div>" +
          '<div class="blog-card__meta">' + esc(p.date) +
            (p.cats.length ? " &middot; " + esc(p.cats.join(", ")) : "") +
          "</div>" +
        "</div>";
      frag.appendChild(a);
    });
    container.innerHTML = "";
    container.appendChild(frag);
  }

  function fail() {
    container.innerHTML =
      '<p class="blog-loading"><a href="' + BLOG + '" target="_blank" rel="noopener">' +
      "Read my latest posts on the blog &rarr;</a></p>";
  }

  fetch(FEED, { credentials: "omit" })
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
    .then(function (xml) { render(fromFeed(xml)); })
    .catch(function () {
      // feed not available — fall back to scraping the homepage
      fetch(BLOG, { credentials: "omit" })
        .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
        .then(function (html) { render(fromHtml(html)); })
        .catch(fail);
    });
})();
