# seikanlp_sematt

This is a tagger program for a specific tagging task.


## Requirements

- SeikaNLP version 0.2.0 or later


## Installation

Clone/download the git repository of this software.

Set seikanlp path to PYTHONPATH
~~~~
$ export PYTHONPATH=/path/to/seikanlp/src
~~~~


## Files and Directories

~~~~
+-- data             ... directory to place input data
+-- log              ... directory to export log files
+-- models           ... directory to export/place model files
|  +-- main          ... directory to export model files
+-- src              ... source code directory
~~~~


## Available Tasks

### Semantic attribute annotation

- Token attribute annotation by rule-based model
    - Given a sequence of tokens (sentence), the model assigns a token-level label (attribute)  
      to each token by outputting a most frequent occurred label of each token.

      $ python src/seika_attribute_annotator.py [--options]

Descriptions of options are shown by executing src/seikanlp.py with --help/-h option.


## License

Copyright (c) 2019, National Institute of Information and Communications Technology  
Released under the MIT license https://opensource.org/licenses/mit-license.php


## Contact

Shohei Higashiyama  
National Institute of Information and Communications Technology (NICT), Seika-cho, Kyoto, Japan  
shohei.higashiyama [at] nict.go.jp
