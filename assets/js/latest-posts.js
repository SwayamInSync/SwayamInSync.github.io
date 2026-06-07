// "Latest Blogs" — pull the newest 3 posts from the Quarto blog at
// swayaminsync.github.io/swayam-script and render them as cards.
// Same-domain in production; works locally too because GitHub Pages serves the
// blog with `Access-Control-Allow-Origin: *`. Degrades to a "visit blog" link.
(function () {
  var BLOG = "https://swayaminsync.github.io/swayam-script/";
  var container = document.getElementById("latest-posts");
  if (!container) return;

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function abs(url) {
    try { return new URL(url, BLOG).href; } catch (e) { return BLOG; }
  }

  fetch(BLOG, { credentials: "omit" })
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
    .then(function (html) {
      var doc = new DOMParser().parseFromString(html, "text/html");
      var posts = Array.prototype.slice.call(doc.querySelectorAll(".quarto-post"));
      if (!posts.length) throw new Error("no posts found");

      posts.sort(function (a, b) {
        return (+b.getAttribute("data-listing-date-sort") || 0) -
               (+a.getAttribute("data-listing-date-sort") || 0);
      });

      var frag = document.createDocumentFragment();
      posts.slice(0, 3).forEach(function (p) {
        var titleEl = p.querySelector(".listing-title a") || p.querySelector(".listing-title");
        var title = titleEl ? titleEl.textContent.trim() : "Untitled";
        var linkEl = p.querySelector(".listing-title a[href]") || p.querySelector("a[href]");
        var href = linkEl ? abs(linkEl.getAttribute("href")) : BLOG;
        var dateEl = p.querySelector(".listing-date");
        var date = dateEl ? dateEl.textContent.trim() : "";
        var imgEl = p.querySelector(".thumbnail img, img.thumbnail-image");
        var img = imgEl ? abs(imgEl.getAttribute("src")) : "";
        var cats = Array.prototype.map.call(
          p.querySelectorAll(".listing-category"),
          function (c) { return c.textContent.trim(); }
        ).slice(0, 3);

        var card = document.createElement("a");
        card.className = "blog-card";
        card.href = href;
        card.target = "_blank";
        card.rel = "noopener";
        card.innerHTML =
          (img ? '<div class="blog-card__thumb"><img loading="lazy" src="' + esc(img) + '" alt=""></div>' : "") +
          '<div class="blog-card__body">' +
            '<div class="blog-card__title">' + esc(title) + "</div>" +
            '<div class="blog-card__meta">' + esc(date) +
              (cats.length ? ' &middot; ' + esc(cats.join(", ")) : "") +
            "</div>" +
          "</div>";
        frag.appendChild(card);
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
