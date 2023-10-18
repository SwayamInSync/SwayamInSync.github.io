# Swayam's Portfolio

## Local Setup

### With Docker
```
$ docker compose up
```
### Legacy Setup (works fast on codespaces)
```
$ bundle install
# assuming pip is your Python package manager
$ pip install jupyter
$ bundle exec jekyll serve --lsi
```

## Deployment

Deploying your website to [GitHub Pages](https://pages.github.com/) is the most popular option.
Starting version [v0.3.5](https://github.com/alshedivat/al-folio/releases/tag/v0.3.5), **al-folio** will automatically re-deploy your webpage each time you push new changes to your repository! :sparkles:

**For personal and organization webpages:**

1. The name of your repository **MUST BE** `<your-github-username>.github.io` or `<your-github-orgname>.github.io`.
2. In `_config.yml`, set `url` to `https://<your-github-username>.github.io` and leave `baseurl` empty.
3. Set up automatic deployment of your webpage (see instructions below).
4. Make changes, commit, and push!
5. After deployment, the webpage will become available at `<your-github-username>.github.io`.

**For project pages:**

1. In `_config.yml`, set `url` to `https://<your-github-username>.github.io` and `baseurl` to `/<your-repository-name>/`.
2. Set up automatic deployment of your webpage (see instructions below).
3. Make changes, commit, and push!
4. After deployment, the webpage will become available at `<your-github-username>.github.io/<your-repository-name>/`.

**To enable automatic deployment:**

1. Click on **Actions** tab and **Enable GitHub Actions**; do not worry about creating any workflows as everything has already been set for you.
2. Go to Settings -> Actions -> General -> Workflow permissions, and give **Read and write permissions** to GitHub Actions
3. Make any other changes to your webpage, commit, and push. This will automatically trigger the **Deploy** action.
4. Wait for a few minutes and let the action complete. You can see the progress in the **Actions** tab. If completed successfully, in addition to the `master` branch, your repository should now have a newly built `gh-pages` branch.
5. Finally, in the **Settings** of your repository, in the Pages section, set the branch to `gh-pages` (**NOT** to `master`). For more details, see [Configuring a publishing source for your GitHub Pages site](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#choosing-a-publishing-source).

If you keep your site on another branch, open `.github/workflows/deploy.yml` **on the branch you keep your website on** and change on->push->branches and on->pull\_request->branches to the branch you keep your website on. This will trigger the action on pulls/pushes on that branch. The action will then deploy the website on the branch it was triggered from.

<details><summary>(click to expand) <strong>Manual deployment to GitHub Pages:</strong></summary>

If you need to manually re-deploy your website to GitHub pages, go to Actions, click "Deploy" in the left sidebar, then "Run workflow."

</details>

<details><summary>(click to expand) <strong>Deployment to another hosting server (non GitHub Pages):</strong></summary>

If you decide to not use GitHub Pages and host your page elsewhere, simply run:

```bash
$ bundle exec jekyll build --lsi
```

which will (re-)generate the static webpage in the `_site/` folder.
Then simply copy the contents of the `_site/` directory to your hosting server.

If you also want to remove unused css classes from your file, run:

```bash
$ purgecss -c purgecss.config.js
```

which will replace the css files in the `_site/assets/css/` folder with the purged css files.

**Note:** Make sure to correctly set the `url` and `baseurl` fields in `_config.yml` before building the webpage. If you are deploying your webpage to `your-domain.com/your-project/`, you must set `url: your-domain.com` and `baseurl: /your-project/`. If you are deploying directly to `your-domain.com`, leave `baseurl` blank.

</details>

<details><summary>(click to expand) <strong>Deployment to a separate repository (advanced users only):</strong></summary>

**Note:** Do not try using this method unless you know what you are doing (make sure you are familiar with [publishing sources](https://help.github.com/en/github/working-with-github-pages/about-github-pages#publishing-sources-for-github-pages-sites)). This approach allows to have the website's source code in one repository and the deployment version in a different repository.

Let's assume that your website's publishing source is a `publishing-source` subdirectory of a git-versioned repository cloned under `$HOME/repo/`.
For a user site this could well be something like `$HOME/<user>.github.io`.

Firstly, from the deployment repo dir, checkout the git branch hosting your publishing source.

Then from the website sources dir (commonly your al-folio fork's clone):

```bash
$ bundle exec jekyll build --lsi --destination $HOME/repo/publishing-source
```

This will instruct jekyll to deploy the website under `$HOME/repo/publishing-source`.

**Note:** Jekyll will clean `$HOME/repo/publishing-source` before building!

The quote below is taken directly from the [jekyll configuration docs](https://jekyllrb.com/docs/configuration/options/):

> Destination folders are cleaned on site builds
>
> The contents of `<destination>` are automatically cleaned, by default, when the site is built. Files or folders that are not created by your site will be removed. Some files could be retained by specifying them within the `<keep_files>` configuration directive.
>
> Do not use an important location for `<destination>`; instead, use it as a staging area and copy files from there to your web server.

If `$HOME/repo/publishing-source` contains files that you want jekyll to leave untouched, specify them under `keep_files` in `_config.yml`.
In its default configuration, al-folio will copy the top-level `README.md` to the publishing source. If you want to change this behavior, add `README.md` under `exclude` in `_config.yml`.

**Note:** Do _not_ run `jekyll clean` on your publishing source repo as this will result in the entire directory getting deleted, irrespective of the content of `keep_files` in `_config.yml`.

</details>

---

#### Upgrading from a previous version

If you installed **al-folio** as described above, you can configure a [GitHub action](https://github.com/AndreasAugustin/actions-template-sync) to automatically sync your repository with the latest version of the theme.

Go to Settings -> Actions -> General -> Workflow permissions, give Read and write permissions to GitHub Actions, check "Allow GitHub Actions to create and approve pull requests", and save your changes.

Then go to Actions -> New workflow -> set up a workflow yourself, setup the following workflow and commit your changes:

```yaml
name: Sync from template
on:
    # cronjob trigger
  schedule:
  - cron:  "0 0 1 * *"
  # manual trigger
  workflow_dispatch:
jobs:
  repo-sync:
    runs-on: ubuntu-latest
    steps:
      # To use this repository's private action, you must check out the repository
      - name: Checkout
        uses: actions/checkout@v3
      - name: actions-template-sync
        uses: AndreasAugustin/actions-template-sync@v0.7.3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          source_repo_path: alshedivat/al-folio
          upstream_branch: master
```

You will receive a pull request within your repository if there are some changes available in the template.

Another option is to manually update your code by following the steps below:

```bash
# Assuming the current directory is <your-repo-name>
$ git remote add upstream https://github.com/alshedivat/al-folio.git
$ git fetch upstream
$ git rebase v0.9.0
```

If you have extensively customized a previous version, it might be trickier to upgrade.
You can still follow the steps above, but `git rebase` may result in merge conflicts that must be resolved.
See [git rebase manual](https://help.github.com/en/github/using-git/about-git-rebase) and how to [resolve conflicts](https://help.github.com/en/github/using-git/resolving-merge-conflicts-after-a-git-rebase) for more information.
If rebasing is too complicated, we recommend re-installing the new version of the theme from scratch and port over your content and changes from the previous version manually.

---

## Posting Jupyter Notebooks as Blogs
1. Convert to Markdown
  ```
  jupyter nbconvert --to markdown your_notebook.ipynb
  ```
  This will save the requried files in a folder say `blog_files`

2. Push that folder inside `assets`
3. Edit the following
```
# replace all the ![](img_path) to <img src="/assets/blog_files/blog_5_1.png">
OR
{% include figure.html path="assets/blog_files/blog_3_1.png" class="img-fluid rounded z-depth-1" %}
```
