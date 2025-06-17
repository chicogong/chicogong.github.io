source "https://rubygems.org"

# Hello! This is a Gemfile for the Jekyll site.
# 
# This file is used to bundle dependencies for the Jekyll site.
# To install dependencies, run:
#   bundle install
#
# To build the site, run:
#   bundle exec jekyll serve

gem "jekyll", "~> 4.3.0"

# The theme of the site
gem "jekyll-theme-chirpy", "~> 6.0", ">= 6.0.1"

# Jekyll plugins
group :jekyll_plugins do
  gem "jekyll-paginate"
  gem "jekyll-redirect-from"
  gem "jekyll-seo-tag"
  gem "jekyll-archives"
  gem "jekyll-sitemap"
  gem "jekyll-feed"
  gem "jekyll-compose"
  gem "jekyll-spaceship"
end

# Platforms
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# For lock file
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby] 