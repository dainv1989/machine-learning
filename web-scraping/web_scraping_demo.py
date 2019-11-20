from selenium import webdriver as wd
from bs4 import BeautifulSoup as bs
import pandas as pd

# Chromium driver download:
# https://sites.google.com/a/chromium.org/chromedriver/downloads
# Firefox driver download:
# https://github.com/mozilla/geckodriver/releases/
# put these drivers to PATH
driver = wd.Chrome("/usr/lib/chromium-browser/chromedriver")

patches = []
links = []
diffs = []

patchwork_ozlabs = "https://patchwork.ozlabs.org/project/linux-mtd/list/"
driver.get(patchwork_ozlabs)

content = driver.page_source

