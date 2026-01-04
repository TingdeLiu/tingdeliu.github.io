# frozen_string_literal: true

source "https://rubygems.org"

# 基础 Jekyll
gem "jekyll", "~> 4.3"

# Jekyll 插件
group :jekyll_plugins do
  gem "jekyll-sitemap"
  gem "jekyll-feed"
  gem "jekyll-paginate"
  gem "jekyll-seo-tag"
end

# Windows 平台需要的 gem
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]

# 性能提升（可选）
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]
