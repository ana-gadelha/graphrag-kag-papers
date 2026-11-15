# Cytoscape.js bundle

`cytoscape.min.js` is the unmodified Cytoscape.js 3.35.0 browser distribution,
licensed under MIT and downloaded from the upstream release tag:

- Bundle: <https://raw.githubusercontent.com/cytoscape/cytoscape.js/v3.35.0/dist/cytoscape.min.js>
- License: <https://raw.githubusercontent.com/cytoscape/cytoscape.js/v3.35.0/LICENSE>
- License text is also included as [CYTOSCAPE_LICENSE.txt](CYTOSCAPE_LICENSE.txt).

SHA-256 of the bundled JavaScript:

```text
71fa9ebec21a7ca2c40cff6aacc30d1f6464e363662ad4b2c197e1adc770cd94
```

The app loads this file locally through Streamlit `components.v2`; it does not
request the library from a CDN at runtime.
