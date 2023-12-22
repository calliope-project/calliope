REPLACEMENTS = {
    # Changelog
    "|new|": "<span class='bubble green'>new</span>",
    "|fixed|": "<span class='bubble blue'>fixed</span>",
    "|changed|": "<span class='bubble yellow'>changed</span>",
    "|backwards-incompatible|": "<span class='bubble red'>backwards-incompatible</span>",
    # Math documentation
    "|REMOVED|": "<span class='bubble red'>REMOVED</span>",
    "|UPDATED|": "<span class='bubble yellow'>UPDATED</span>",
    "|NEW|": "<span class='bubble green'>NEW</span>",
}


def on_page_markdown(markdown, **kwargs):
    for old, new in REPLACEMENTS.items():
        markdown = markdown.replace(old, new)
    return markdown