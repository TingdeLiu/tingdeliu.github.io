# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a personal academic blog and research website built with Jekyll and hosted on GitHub Pages. The repository contains two template directories (`KwanWaiPang.github.io-main` and `chirpy-starter-main`) that serve as reference implementations, with the actual site configuration being in the root or one of these directories.

## Jekyll Site Architecture

### Primary Configuration Files

- `_config.yml`: Main Jekyll configuration for site metadata, plugins, navigation, and build settings
  - Configures site title, description, social links, and analytics
  - Defines kramdown settings for Markdown processing with MathJax support
  - Manages pagination, permalinks, and Jekyll plugins

### Directory Structure

**Content Directories:**
- `_posts/`: Blog posts in Markdown format, must follow naming convention `YYYY-MM-DD-title.md`
- `_layouts/`: HTML templates that define page structure (default.html, post.html, page.html)
- `_includes/`: Reusable HTML components included in layouts
- `_sass/`: SASS/SCSS stylesheets for site styling
- `_tabs/`: Special pages like About, Archive, Tags (Chirpy theme specific)
- `_data/`: YAML data files for navigation, contact info, etc.
- `assets/`: Static assets including images, JavaScript, and CSS

**Template References:**
- `KwanWaiPang.github.io-main/`: Reference implementation with custom styling and blog examples
- `chirpy-starter-main/`: Clean Chirpy theme starter template

### Blog Post Front Matter

All blog posts must include YAML front matter:
```yaml
---
layout: post
title: "Post Title"
date: YYYY-MM-DD
tags: [tag1, tag2]
comments: true
author: authorname
toc: true
excerpt: "Brief description for preview"
---
```

## Development Commands

### Local Development

**Build the site:**
```bash
bundle exec jekyll build
```

**Serve locally with live reload:**
```bash
bundle exec jekyll serve
```
The site will be available at `http://127.0.0.1:4000/`

**Build for production:**
```bash
JEKYLL_ENV=production bundle exec jekyll build -d "_site"
```

### Testing

**Validate HTML output:**
```bash
bundle exec htmlproofer _site --disable-external --ignore-urls "/^http:\/\/127.0.0.1/,/^http:\/\/0.0.0.0/,/^http:\/\/localhost/"
```

### Dependency Management

**Install dependencies:**
```bash
bundle install
```

**Update dependencies:**
```bash
bundle update
```

## Deployment

The site uses GitHub Actions for automated deployment (see `.github/workflows/pages-deploy.yml`):
- Triggered on push to `main` or `master` branches
- Builds with Ruby 3.3 and Jekyll
- Tests HTML output with htmlproofer
- Deploys to GitHub Pages automatically

Manual deployment is not typically needed as GitHub Actions handles this.

## Key Themes and Plugins

**Jekyll Theme:**
- Uses `jekyll-theme-chirpy` (version ~> 7.4) or custom theme based on the configuration
- Chirpy provides a responsive, feature-rich blog theme with dark mode, search, and TOC

**Essential Plugins:**
- `jekyll-sitemap`: Generates sitemap.xml for SEO
- `jekyll-feed`: Creates Atom feed
- `jekyll-paginate`: Enables post pagination
- `jekyll-archives`: Manages category and tag archives (Chirpy)

## Content Creation

### Creating New Blog Posts

1. Create a new file in `_posts/` with format: `YYYY-MM-DD-title.md`
2. Add required front matter (layout, title, date, tags, author, toc, excerpt)
3. Write content in Markdown with optional table of contents: `* 目录\n{:toc}`
4. Commit and push - GitHub Actions will deploy automatically

### Markdown Features

**Math Support:**
- MathJax is configured for LaTeX equations
- Inline: `$equation$`
- Block: `$$equation$$`

**Code Syntax Highlighting:**
- Uses Rouge syntax highlighter
- Fenced code blocks with language specification supported

**Image Insertion:**
```html
<div align="center">
  <img src="image-url" width="60%" />
<figcaption>Caption text</figcaption>
</div>
```

## Configuration Notes

### Dual Configuration Setup

This repository appears to have two parallel configurations:
1. Custom Jekyll setup (in `KwanWaiPang.github.io-main/`)
2. Chirpy theme starter (in `chirpy-starter-main/`)

When making changes, verify which configuration is active by checking:
- The root `_config.yml` settings
- GitHub Pages settings in repository Settings > Pages

### Important Settings

**Permalink Structure:**
- Posts: `/posts/:title/` (Chirpy) or `/:title/` (custom)
- Paginate path: `/home/page:num/` or `/page:num/`

**Timezone:**
- Set to `Asia/Beijing` - update based on your location

**Base URL:**
- Typically empty for user/org sites: `baseurl: ""`
- For project sites: `baseurl: "/repository-name"`

## Working with Reference Templates

The `KwanWaiPang.github.io-main/` directory contains:
- Extensive blog post examples covering SLAM, robotics, AI research
- Custom layouts and styling
- Template file: `yyyy-mm-dd-blogName.md` for creating new posts

The `chirpy-starter-main/` directory contains:
- Clean Chirpy theme implementation
- Minimal configuration example
- DevContainer support for consistent development environment
