commit 4d0958e9d4da4b7509fe3c588336991f4977e794
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Jan 16 14:58:24 2019 +0000

    Add epsilon-cost, epsilon-carbon, and cost of carbon constraints

commit 7fe0234ff9ae5fec10887c2ec08fafc121dbb247
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Jan 16 14:03:22 2019 +0000

    timeseries dimension fixes

commit e3d34700ee3cc75ecb31c988c0487a17a9d49208
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Thu Sep 13 17:13:48 2018 +0100

    add missing scenario dimension to constraint

commit 8584d72d75ef31a6714a30a82f0fd204c7e283a9
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Sep 3 17:43:05 2018 +0100

    Minor fixes

commit b9269df9dc1f7a02c23fcf1d4176318516a662b6
Merge: c74a53b 94d5727
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue Jun 19 15:14:03 2018 +0100

    merge from upstream master

commit 94d57271d10415c0c71d4bbc6678c4aff79d0624
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Sat Jun 16 20:59:04 2018 +0200

    Add `excess_supply` variable when ensuring feasibility

commit 5b0cb2e0f80121a2bbb1d01461ba481819df9cc2
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Jun 13 15:54:56 2018 +0200

    Alphabetically sort `model` and `defaults` config YAML files (#122)

commit 93e9d1a25ec8ddf03899dc03a1418e202b98ab13
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jun 13 12:07:58 2018 +0200

    required_constraints must also be allowed_constraints (#120)

commit c74a53b835a3aa10e3bfe1e82c74462d0b576541
Merge: 6d5bd05 5d29a7f
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Jun 13 09:23:17 2018 +0100

    Merge branch 'master' into stochastic_optimisation

commit 5d29a7f9a2d45c04cd908aed832be6f6e6affe16
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Jun 13 10:16:24 2018 +0200

    Fix systemwide constraints to include transmission techs (#119)

commit 6d5bd054c9740e840f21df7e3a7cf074e08dd036
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue Jun 12 15:10:09 2018 +0200

    Update `get_param` of new `unit_cap_systemwide` constraint

commit 12132db8a271e39d9df0fe24419865fb389a67a1
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue Jun 12 14:31:37 2018 +0200

    Fix missing `scenario` dimension in `carrier_con` index

commit 65597f6ed27414faa3188f4d2210541880fdcdeb
Merge: 67a125e f187f1f
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Jun 11 16:37:59 2018 +0100

    Merge branch 'master' into stochastic_optimisation

commit f187f1fbaa31a66f98d976cc4b4327f4a7342234
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Jun 11 16:17:13 2018 +0200

    Add `--mem` option to SLURM run generation script

commit af939f429d61cac424a8b58e294606be829cacbe
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Jun 11 12:42:14 2018 +0200

    Add SLURM to script generator (#118)

commit 6454a9774a6f70f8419f03e77741bbe0fae2378f
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Jun 11 10:47:09 2018 +0200

    Post-release commit, update to 0.6.3-dev

commit 67a125e1956edda69ad20726d9ad8ab12e227039
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Jun 11 09:21:24 2018 +0100

    Fix bug when running without scenarios from cli

commit 9ca95f351b05df962e31ca2508883bf0550a5748
Merge: e895d6d 8ba6abd
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue Jun 5 11:46:28 2018 +0100

    Merge updates from master

commit 8ba6abd2330600a9135b20f195b612670e3e4bba
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Jun 4 12:08:20 2018 +0100

    Release v0.6.2

commit 4c6e4ea29b558a37c9073a33798ffc0728a6c6e5
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Jun 4 12:06:17 2018 +0100

    Update doc plots for new release; minor changelog updates

commit 8cbbe730acd7488d5569d4e3818257047874efb3
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Jun 4 11:36:02 2018 +0100

    Update where we store & check for FutureWarnings and DeprecationWarnings (#117)
    
    Fixes issue #108

commit 3cb29ee6d42918c449f81da6fbbcbd72453c4487
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri Jun 1 12:49:25 2018 +0100

    Minor import order update

commit bff2f35bdeca6b40e42791046bd4186b7d8b9e5d
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri Jun 1 11:33:43 2018 +0100

    Add unit_capacity_systemwide_constraint (#116)

commit 53b2bc0e93b716aa2e93da3951c685424bd6ab48
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jun 1 12:22:53 2018 +0200

    Fix negative om_con costs in conversion techs (#109)

commit b8592cf2f5aa5253964197ed042e21966381c5a0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jun 1 10:52:59 2018 +0200

    Minor doc updates

commit d7e92f6429f15588415922b9765847ba6d9b5e0e
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Thu May 31 19:41:59 2018 +0100

    Add docs and FutureWarning for cyclic storage; minor cleanup

commit e895d6dbfa4bae31b09844f8a7ec3e7e6cecc065
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue May 29 18:51:24 2018 +0100

    Ensure floating point error doesn't break sum(probability) == 1

commit 193e6bea2b4e8597417b0c3aac49d9e64e4e70ff
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed May 30 19:22:06 2018 +0100

    Remove need for scenario dimension on all timestep vars; update objective

commit 52f544464fa8357eaf888beb03bb3687472266ef
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 30 17:13:11 2018 +0200

    Cosmetic fixes
    
    * Docs: add GitHub ribbon to index page
    * CLI: use ipython debugger instead of built-in pdb

commit 8a3df6d7c14db28e6a1df9c11534526f73c0a89d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 30 13:18:56 2018 +0200

    Update changelog; fix a trailing bracket

commit 0455ec4f26102410a5e4c726a8c2f9e578cc4176
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 30 12:47:15 2018 +0200

    Allow YAML comments to pass through AttrDict

commit 2cc849870c165a457d8261ce326412521f6be35f
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed May 30 12:21:26 2018 +0100

    Separate `primary_carrier` into `primary_carrier_in`/`_out` (#111)
    
    Separate `primary_carrier` into `primary_carrier_in`/`_out` for conversion_plus techs, to allow `om_con` and `om_prod` costs

commit 9deaae1553544e16609f5f0a7f98485aee79847b
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue May 22 08:16:44 2018 +0100

    Update error raised by multiple scenarios in plan mode

commit 7e980441d81c81c7aad5e44e0cd8f68c03416e6b
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue May 22 07:48:39 2018 +0100

    `robust_plan` -> `scenario_plan` mode name

commit c46c081c100c1963bfa91609f1bcb4e84c52274c
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon May 21 15:21:14 2018 +0100

    Update get_param in Pyomo backend

commit d0afa0e34b5b2f609b28b889eec5715218e73549
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon May 21 15:20:49 2018 +0100

    run_scenario in cli; keep results of new scenarios; fix unsorted lists

commit fc7cfe858935f062e023f5ce1179e467a692b0e9
Merge: b812e83 6348b84
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed May 9 19:06:42 2018 +0100

    Merge changes from current master

commit 6348b842c1188e9322de30b1ec63221bbd072c50
Author: Katrin Leinweber <kalei@posteo.de>
Date:   Tue May 8 13:51:29 2018 +0200

    Hyperlink DOIs against preferred resolver

commit 0e53f141c64b33225d2a4ae7f19539b285f0aa08
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri May 4 18:55:42 2018 +0200

    Add constraints & variables to track storage between clustered datesteps (#106)
    
    Also adds cyclic storage, which (if enabled) links the storage at the end of the whole (unclustered) timeseries to the storage in the initial step.

commit fbb3097b214c463374d0e162041d0fcaaf5d2b4f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 4 17:06:28 2018 +0200

    Plotting fixes
    
    Allow plotting for:
    
    * inputs-only models,
    * single location models,
    * models without location coordinates

commit 1bc03e5b93ff27e35d7d50199ab990e5d906a703
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 4 13:56:39 2018 +0200

    Fix CLI error when running minimal models

commit 91959ec3ea2cf33fe4359509bcddf461ed1da4da
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 4 07:26:05 2018 +0200

    Tagline cleanup

commit 5663309875b317673aceb8490a11bb4464c3f3ab
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 4 07:03:05 2018 +0200

    Update release checklist

commit 197216e0625685fd199778e00902e33c052c8b5d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 4 07:01:02 2018 +0200

    Begin v0.6.2

commit 60894bb5781d265b42d064290c1a3006d1fd71b7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 4 06:53:23 2018 +0200

    Release v0.6.1

commit c41c5bc9106324683b69a83fd22d73f4db1a5f41
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 3 18:11:22 2018 +0200

    CLI: fix save_plots; tests: catch all warnings

commit 971bab9503d75bbf4b6648b6e0ee3e148c370f81
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Thu May 3 17:47:28 2018 +0200

    Ensure pure dicts can be used to create calliope Model

commit 23f997c2caf31a375fac56b29bec937d208cdad6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 3 17:42:25 2018 +0200

    AttrDict: fix union with all-empty subdicts

commit 159664ba3623ea0134ed473836f296b4485547c8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 2 16:48:09 2018 +0200

    Minor fixes to JOSS paper

commit 268cad1d8e298640944ea13b0e983bdb100a104b
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue May 1 11:02:48 2018 +0200

    Added functionality to allow user-defined clusters in timeseries aggregation (#105)
    
    Addresses issue #90

commit b812e834b078fc25b4228e3e0c3027040406dec7
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue May 1 09:59:41 2018 +0200

    update get_param & allow running single scenario

commit 9faef10cc772dad33ede0888cb95a47d3f1dc9f0
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue May 1 08:37:15 2018 +0200

    Update terminology; revert objective formulation

commit 31afdb7f6a7428fd0268bf48716fd1ca0ae6a118
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Apr 30 13:22:39 2018 +0200

    Add JOSS paper

commit d412415466e471c5f2b0fbf1fd5703f3fd1853b2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Apr 30 11:23:18 2018 +0200

    Clean up AUTHORS and contribution licensing info

commit a85df3b11ab73c859c6cc6303d59dd7609825c87
Author: Graeme Hawker <graeme.hawker@strath.ac.uk>
Date:   Fri Apr 27 18:30:44 2018 +0200

    Objective function cost class/sense definition in run config (#103)
    
    * Allow objective function cost class and sense to be defined in run configuration
    
    * Reduced objective tests to single days, added test for unused objective options warning
    
    * Moved new test under test_core_preprocessing.py::TestChecks::test_unrecognised_model_run_keys
    
    Signed-off-by: Graeme Hawker <graeme.hawker@strath.ac.uk>

commit 90c96e88a85794f914da8fa8f69ef02e56ecc0bb
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 27 18:10:26 2018 +0200

    core: better name and docs for save_commented_model_yaml

commit 2275e655d65dd2d558311663c00050d27efe1d74
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 27 17:34:21 2018 +0200

    Contribution and developer guide updates

commit 0a9ec5a21db9eefed525e5707b484427166486c6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 27 15:53:20 2018 +0200

    io: fix warning on version mismatch

commit 08b6b71073010732a24a18ff064897d57833f289
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 27 15:53:06 2018 +0200

    plotting: layout_updates & plot_kwarg_updates args

commit 2c5d451279dbb9c6ce1e989c748ea1429231b9ff
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri Apr 27 14:01:31 2018 +0200

    Update pull request template

commit 5919561d86253e94105b3e80901577c6d4a19871
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 26 16:04:46 2018 +0200

    Format as list when printing warnings and errors

commit 9c0e238fa5b7cc6e6f53be1d497413eef7e3eaa2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 25 14:56:23 2018 +0200

    Update calliope_version in example models

commit 9ae96a544544f23dcf49c1a423992ab87d1005cb
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 24 15:10:41 2018 +0200

    README as long PyPI description; Fix doc contributors

commit 5b3b9d396e993bfcae502500a858f3c627f63255
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri Apr 20 14:23:41 2018 +0200

    Begin 0.6.1-dev

commit 4a9434f9982cabf0d151927730e7702e50554b42
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri Apr 20 10:02:42 2018 +0200

    Release v0.6.0

commit 94512c15ec45a3c88a4fb79136bb07c285000dd2
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri Apr 20 09:57:12 2018 +0200

    Update development docs

commit af65b79818f77336b1c530f01e56cb0508efdde6
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri Apr 20 09:54:41 2018 +0200

    Update plotting tests; re-build doc plots and notebooks

commit 52b7683489991c97d34db3ccf0fb8cef97a2f70d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 20 07:30:28 2018 +0200

    Doc fixes

commit c0a52093e4cd3ed6b9f0e6aeb72de0e862171be2
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Thu Apr 19 22:35:19 2018 +0200

    Update saving plots to file

commit d26f004d704046ce7056855cf1fad36ffee88f97
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 19 22:15:34 2018 +0200

    Fix setup.py

commit f40e2f17bae87c4dbdac689d2d1400dce33a4279
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 19 22:08:33 2018 +0200

    Update requirements and setup.py

commit 5779934a0c012b7dba25efb3e6bd24338cb0571a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 19 21:48:25 2018 +0200

    Minor doc updates

commit b2d784df830cf8611693d28fd434d2ed2658b819
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Thu Apr 19 18:38:06 2018 +0200

    Update docs

commit fcc30231e93718dce828b6d3fd6321fe99fdb709
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 19 17:57:43 2018 +0200

    Add info() method to core.Model

commit 42feb520073add483331db09124ef0c92405e40e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 19 17:43:37 2018 +0200

    Minor fixes to docs, don't base_tech-group capacity plots

commit f499dadf2764d803935cbb73cfbca1d69fcaf20e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 19 16:48:45 2018 +0200

    Add to docs, fix missing comma

commit 3859e36323fdef5c58345f6adee8bdc337f033d8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 19 16:37:30 2018 +0200

    Minor improvements
    
    * plotting: use natsort for better sorting in plots
    * plotting.timeseries: unless tech-subsetting,
      sort techs in the stack by using xarray.var
    * plottnig: raise ValueErrors, not ModelErrors
    * core.model: add labels to log_time calls
    * convert: remove _per_distance top-level items
    * clean up tests

commit aabb8c710ede541e944a49005bcea5a68e607cdd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 19 13:06:51 2018 +0200

    Minor fixes
    
    * plotting.timeseries: properly group unmet_demand
    * conversion: remove per_distance
    * core.model: do not ask to overwrite empty results

commit 37d5574eed9783db16a81d8af96076e7319bcb9e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 19 13:05:48 2018 +0200

    Fix reserve_margin

commit 105da14c306046a0a12a2ccd3d3386a529ec00f0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 18 11:57:50 2018 +0200

    Plotting functions cleanup
    
    * Move hardcoded monetary costs default into summary plot only
    * Improved plotting argument processing and docstrings
    * Use check_error_or_warning everywhere in tests

commit a5ce4180ba213e9010ebffbeb1e4a5aef796ff17
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 17 18:40:23 2018 +0200

    Default to monetary cost class in plotting

commit 07f93ed13429afadbfd62f0c1aca18add99556ae
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue Apr 17 17:41:52 2018 +0200

    Fix plotting timeseries in models w/o supply_plus

commit 6527378a2cab09eedaa8e6016db88ea2feb86a75
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Apr 16 09:21:13 2018 +0200

    Removed ignored directory ".ipynb_checkpoints"

commit 0cc6f886eefba65685800e97154fce0a003d5089
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Apr 16 09:18:56 2018 +0200

    Fix resource_con decision var domain; update plots

commit cb99deff2ad9020813e533d99b7025dfbdf78b22
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri Apr 13 22:38:54 2018 +0200

    Update plots and tutorials

commit 7908b65e38c369802708ec60ab1a7a7029d2b0d4
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri Apr 13 21:41:58 2018 +0200

    logging -> logger

commit d7d4d2a9f039b5d54aa81c39c58b42b2ad39f5b1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 13 18:05:06 2018 +0200

    Add --quiet option to CLI

commit aa071eea476b899866ebc82dfc364dff09a5f557
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 13 17:20:58 2018 +0200

    Minor cleanup
    
    * Convert: enable ensure_feasibility in models that had unmet_demand
    * Clean up package list in installation docs
    * Add projects to CONTRIBUTIING and dev docs

commit a4e109dcb46b8eabba12cdc36510cb1381f41628
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 12 17:58:49 2018 +0200

    Add summary plot HTML to docs

commit edf2fdf6bc0321c9fab01c338617b796bbd80fed
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Thu Apr 12 17:51:25 2018 +0200

    Add tests

commit 728cb912db21e4a1c70563378fcc9873394cb306
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 12 17:43:15 2018 +0200

    Check for more unrecognised settings; fix model.name

commit fbfc3136566f70d0664591b469a2d68981bf8d0f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 12 17:03:01 2018 +0200

    Fixes to constraints, preprocessing, conversion
    
    * Correctly deal with ``techs.foo: None`` when merging multiple
      location definitions
    * Tech name defaults to tech ID if not set
    * Charge_rate is optional for storage techs
    * Transmission techs may define energy_cap_scale
    * Infinite capacity upper bound correctly checked for
    * Do not warn when plotting before running
    * Update conversion map and fix minor conversion
      script issues

commit 87cfda4e2575b52165a38f8ee9acd59969d44fce
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 12 16:10:52 2018 +0200

    Warn on plotting non-optimal models (don't plot in CLI)

commit d7e35ba47da0036a971fea50adcd9e30835aba08
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Thu Apr 12 12:02:39 2018 +0200

    fix tests

commit 270e74769998951b050dcfcf1e03c27b2c1bb5f6
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Apr 11 23:37:12 2018 +0200

    Improve logging functionality

commit 637e2dd7a619481828a660f3d6b3b154e0052bbf
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 11 18:44:09 2018 +0200

    Cleanup
    
    * Updated requirements
    * Things that should not be in __init__.py
    * Other minor cleanup

commit 98031695544020424c085f83cc229b43e21552de
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Apr 11 18:21:22 2018 +0200

    Update package requirements

commit 1d956b0da7a5033915a276187b5de5e79387a582
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 11 17:56:53 2018 +0200

    Docs: add supply_plus details, fix extreme time func

commit 5b611804f07c0ecffc3916b897aeb910f1cd6559
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Apr 11 17:04:29 2018 +0200

    update time clustering funcs; add tests; update storage plotting
    
    * Time clustering now only uses scikit-learn, removed scipy dependency.
    * 'maxdist' hierarchical clustering method no longer possible
    * storage timeseries plot shows charge and discharge seperately
    * Added tests for time masking/resampling/clustering with non-hourly timeseries

commit aaeddf94f8fdeb415363d909cf16510549db7d06
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Apr 11 16:52:53 2018 +0200

    Add tests; fix some spelling

commit f9446b94beee6e411338f8b17632678c90fe2b3c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 11 16:14:44 2018 +0200

    Update SVG figures

commit 648020b99250b9f22278cd0415f712aa8c5f381b
Merge: 150af66 f5afa40
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 11 08:32:41 2018 +0200

    Merge pull request #97 from GraemeHawker/master
    
    Sub-hourly time masking fix compatible with refactored functions

commit f5afa40b56dd5eaad67aeecd5a8498a1376775b2
Author: graeme <graeme@lutralutra.com>
Date:   Wed Apr 11 00:11:19 2018 +0100

    Fixed time masking to be compatible with refactored mask functions
    
    Signed-off-by: graeme <graeme@lutralutra.com>

commit 51c8925337dfcb55356c22ef9d3a9425c5cec474
Author: graeme <graeme@lutralutra.com>
Date:   Tue Apr 10 17:30:12 2018 +0100

    Syncing fork with upstream
    Signed-off-by: graeme <graeme@lutralutra.com>

commit 8e540821058e5bff05baecf8101405abaa36b7fd
Merge: 24e9087 150af66
Author: graeme <graeme@lutralutra.com>
Date:   Tue Apr 10 17:27:35 2018 +0100

    Merge branch 'master' of https://github.com/calliope-project/calliope

commit 150af665e502f905513f43b1717e8b894cecb4bf
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 10 16:36:05 2018 +0200

    Add clustered example timeseries plot

commit 118c5efad4c842e7dbef4c81bc8c7199ab33bddb
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 10 15:43:09 2018 +0200

    Update gitignore to ignore built notebook HTML

commit c63f6dddee63d110b82240fb5d756ab3e4d9ccb2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 10 15:42:17 2018 +0200

    Update changelog and whatsnew

commit 637007f9bd445428cf02b1b4677a5fd2121173c1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Apr 9 19:58:10 2018 +0200

    Fix tutorial NB links, remove built NB HTML files

commit 068caa7a3086e4bcb0632245e71235ae4f7c2eaf
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Apr 9 19:44:46 2018 +0200

    More constraint docstrings

commit d48aa16f641f497a5d522b2f7181edadc09b6230
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Apr 9 19:44:31 2018 +0200

    Better tutorial notebook integration

commit e2e709b54dfd0b807ef9d817f714d485487138a9
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Apr 6 17:33:12 2018 +0200

    Fix tests
    
    They were only broken because warnings were no longer being thrown,
    thanks to removal of empty `distance` and `lookup_remotes` from model_data

commit 24e9087e4c5ba6205b11e0f619339c76b371c455
Merge: a1d16ba bb51577
Author: graeme <graeme@lutralutra.com>
Date:   Fri Apr 6 15:00:47 2018 +0100

    Merge branch 'master' of https://github.com/calliope-project/calliope

commit bb51577924e9aa01244d79f8c41643dec34a7bc0
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Apr 6 15:46:24 2018 +0200

    Move tutorials to _static

commit a1d16ba2eb9629463261f7c71b72a88726eed454
Merge: 0c93d93 d52f86b
Author: graeme <graeme@lutralutra.com>
Date:   Fri Apr 6 14:08:25 2018 +0100

    Merge branch 'master' of https://github.com/calliope-project/calliope

commit 0c93d932aaac1f0b3fa0fcd6c1a4b4670523c4c5
Author: graeme <graeme@lutralutra.com>
Date:   Fri Apr 6 14:06:06 2018 +0100

    Amends time masking to remove hourly timestep requirement, fixes #81
    Signed-off-by: graeme <graeme@lutralutra.com>

commit d52f86b59667db46c8b8697e77b2c99ecf127c6d
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Apr 6 15:04:47 2018 +0200

    Update doc tutorials; remove distance from model_data if empty

commit 35c3bebf7fc548b21b3934be8e6a35f53b268324
Merge: 2ea8657 5a0d551
Author: graeme <graeme@lutralutra.com>
Date:   Fri Apr 6 14:00:50 2018 +0100

    Merge remote-tracking branch 'upstream/master'
    
    # Conflicts:
    #	calliope/_version.py
    #	calliope/analysis.py
    #	calliope/constraints/base.py
    #	calliope/core.py
    #	calliope/read.py
    #	calliope/sets.py
    
    Signed-off-by: graeme <graeme@lutralutra.com>

commit 5a0d5519b911980e2ca02af08d588f45148ffd3c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 6 11:39:08 2018 +0200

    Clean up aux files

commit 9613da23d675659d2e2f2b402d9e8c6b18e8b56a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 6 11:01:35 2018 +0200

    Drop NaN values in to_csv by default
    
    Fixes #83

commit 63422e404a115252742adcd0eefd7c55713c2da9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 6 11:00:04 2018 +0200

    Improve list formatting in AttrDict.to_yaml

commit cd4695e51875a52fe659a3d49cadfdf8d667b7be
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Apr 5 18:28:24 2018 +0200

    Add tests; Add docstrings; Start tutorial notebook updates

commit 7d78ca5f93d267b86cc1c7ab4fb7c65817bc6f93
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Apr 4 20:09:21 2018 +0200

    Update tests & rename git master requirements file

commit 2b5f17dd9cfde8cc2df5ae18b53fcbaf6a308853
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Apr 4 19:48:09 2018 +0200

    Add tests

commit 2af97a27e7c64109972478dcac359751de688ca7
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Apr 4 18:43:39 2018 +0200

    Minor fixes;
    
    * Update SVG fonts to a priority list, starting with Open Sans, ending with sens-serif
    * Add alternate background colours to distringuish clusters in clustered timeseries plots
    * Add multiple tests
    * Add checking for overrides, to provide info on overrides
    * Add scale_clusters to clustering funcs, to allow scaling clustered data to input mean/max/min/sum/None (obligatory mean scaling before this)

commit 85fe56fd4d226f2339f3945b9e14e79d1f720c8e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 4 17:38:04 2018 +0200

    Docs: minor changes to conversion_plus

commit b99b4d02a52ca97e70b73c6b974d102c41cb76f2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 4 10:10:12 2018 +0200

    Docstring and doc updates

commit a7a7980edf2dc9bd45e78567adb7ecd256ef2bb7
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Apr 3 23:40:36 2018 +0200

    Add plotting tests; fix use of multiple cost types

commit 0bdec8ee30741d603e620cd511d0a0abccac56c3
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Apr 3 19:40:44 2018 +0200

    Updated clustering & tests
    
    Still unsure whether clustering is acting as it should, as mean/closest remain diverged in results

commit d8652fcc4a4c26630105d0e644f249a41450ed1a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 3 18:03:09 2018 +0200

    Minor doc updates

commit 9cb87984170c3d01cecab1995f58b12f6d76fe80
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 3 14:13:46 2018 +0200

    Fixes to time processing and additions to tests
    
    * Add barebones clustering func tests
    * Simplify and test time.funcs.drop()
    * Remove need for temporary daily_timesteps attr
    * Hardcode current state of closest and mean clustering in example
      model tests
    * Test --debug CLI option
    * Module-scope fixtures to speed up tests

commit af58eb11e7bdedda221b751ad052afecff3672fc
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Mar 29 22:41:38 2018 +0200

    Update objective; minor plotting fixes

commit 56c0569cd1b61fc32e796f5b79d223d8e77990cd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 29 14:13:11 2018 +0200

    Better conversion; test it

commit ec62b24e52c4a4bc2a5fc4c582d980e29cb7c028
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 29 14:11:59 2018 +0200

    Mark clustering tests as known fail for now

commit 3d97178167262795cbf5809e786836d06b046d88
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Mar 28 16:07:17 2018 +0200

    Add unmet_demand to objective

commit ac17eee82e2e3c3fa3138bf9a82afb9a475b1bf9
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Mar 28 15:26:03 2018 +0200

    Fix PLOTLY_KWARGS update issue

commit 3fc4dd4cff0e8a35cf37e53e0063310c612e9cef
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Mar 28 14:29:07 2018 +0200

    Add RO functionality to calliope 0.6.0 release candidate
    
    TODO:
      * Fix existing tests
      * Add tests for RO
      * Document RO
      * Allow scenario dimension in plotting
      * Add scenario dimension to everything, including non-timeseries params/variables
      * Add running all/some scenarios in parallel/series to CLI

commit fb0a9053730da015143abd397bfa52b22c95308e
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Mar 28 09:07:30 2018 +0200

    Purchase cost / 2 for transmission techs

commit 4352da86001f7cbff43b63eef89dc4c428aa92f8
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 27 16:24:32 2018 +0200

    Fix saving to SVG option

commit 0387f926a4f108271e57ee303fddddf7dd623a17
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 27 15:42:56 2018 +0200

    CLI: permit running model from pre-built NetCDF

commit f6edfdfa4baee7ad0b258270279518147540af27
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 27 15:35:00 2018 +0200

    Minor fixes
    
    * Fix plotting clusters
    * Fix `solver_options` being carried through to solver
    * Fix NaN values caused by infinite capacity in purchase MILP constraints

commit b61d7592146f7071c32893cf655749f97dee5128
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Mar 26 23:14:25 2018 +0200

    Handle clustered data in plotting; fix k-means 'mean' cluster method
    
    And add line break (> 30 chars) for legend items in plotting

commit f7a7cd0edc75cb01771348eee4b92cae7a5801cd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 26 18:15:52 2018 +0200

    Docs: warn about timestep weights, raw requirements URL

commit 80677db5ead8d03cdfb430dd718a2860bc6f85c8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 26 16:47:57 2018 +0200

    Postprocess: fix LCOE and CF when weighting time

commit 7e38227189d0937c087125643a207f4230f5b938
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Mar 26 09:47:58 2018 +0200

    Update checks to avoid memory-error inducing feedback loops

commit 41d26d33f54a85330c633106704b4dca4c860073
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Mar 23 18:25:46 2018 +0100

    Update tests; add checks
    
    * Start checking upper/lower bounds of constraints built in pyomo
    * Remove unnecessary complications existing in storage_cap constraints
    * Add tests for model_data checks
    * Update CSV reading method, to better catch errors

commit dc1293d5f13b5ca078addb5f2ef024261f829e44
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 23 16:04:01 2018 +0100

    Docs: do not yet convert images

commit 139826795c15fbd7f3abad607aa4fe83fe364fbb
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 23 15:59:53 2018 +0100

    Fix group share constraints in absence of supply_plus

commit e2a6295b530d2853f644a5c3a741bbea0213a877
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 23 14:36:31 2018 +0100

    Implement 'calendar_week' padding in masks

commit 4d16dc411833b15856c30eb2bf8277f8b213aa53
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Mar 23 13:49:18 2018 +0100

    Update time masking doc strings; Add time masking example model

commit ca59af2cad801daa56fb2b7877eaf31e58cf509c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 23 13:45:09 2018 +0100

    Test and fix group_share constraints

commit 6727d063fc7f32f941b7284a4f9523239801ffa3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 22 17:44:24 2018 +0100

    Fixes to docs
    
    * Always prefer PNG over SVG in HTML builds
    * Better helper script organisation
    * Minor corrections

commit dae195d96319ffa464429bc3cddc207d33f99bab
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 22 17:40:52 2018 +0100

    Test improvements
    
    * Test resampled and clustered national-scale models
    * Fix and test add_max_demand_timesteps

commit d07fd650deb8cb3763db06233102cf6b7d6ac990
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 22 16:57:31 2018 +0100

    Further test improvements

commit 6307ab3155459c3cf594569cf0ba11c29e9ac2e8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 22 14:37:36 2018 +0100

    Add debug, cli, and io tests

commit 7c2c1f3cb13d38af43641ba98d3bd620b60a9acd
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Mar 21 22:11:05 2018 +0100

    catch rogue uses of `model.mode`

commit a6f8eae1b5e7284c75a22e52357c594621e7c0d1
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Mar 21 20:15:25 2018 +0100

    Add rerun to Pyomo interface & document/test
    
    + Add check_feasibility Pyomo objective

commit 2a1257e11464f846c42e38f94ef153b419d16c82
Merge: a1d7ee6 3d1e3d2
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Mar 21 17:34:30 2018 +0100

    Merge branch 'master' of https://github.com/calliope-project/calliope

commit a1d7ee65552ca554a41c04a26b392d8355264a7f
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Mar 21 17:34:21 2018 +0100

    Update figures

commit 3d1e3d2f0fb7d51e74e5c5e95c3b8a292f2eeb67
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 21 17:00:45 2018 +0100

    Docs: include Open Sans font for SVG figures

commit 7514079da0d69b9848b3746561c12a698f1561ab
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 20 16:00:18 2018 +0100

    Fix tests, change plot hovermode; update master branch install method

commit 1a84522faba6af8934b7b63c94617f6818b51138
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 20 14:57:36 2018 +0100

    Fix API; add help to make.bat; update docs plots; fix operate test

commit 88887a10adff3e0bbe0ef32885d7488a4bc1cb95
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 20 12:30:05 2018 +0100

    Develop docs: do not forget to build plots

commit 7bdfa22234e62823a0a26b6eebe8c26225193550
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 19 18:53:46 2018 +0100

    IO: convert boolean attrs to ints in save_netcdf

commit 72c9906bef5e9e3a64ec741fbbfc92b5050d3c20
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 19 18:52:51 2018 +0100

    Backend: allow build_only=True for debugging

commit b72866abd3729555fa049ae8e531e3986519c9a1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 19 18:51:56 2018 +0100

    Fix capacity plot variable ordering

commit 38365901884c4b9c370a066d54f52371a06fe1c3
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Mar 19 17:56:16 2018 +0100

    Update tests; minor fixes
    
    * Add pyomo backend tests. TODO: only generate backend in constraint checks, don't run
    
    * Fix operate mode termination details
    
    * Add om_con to allowed costs for conversion tech
    
    *Add `operate` to example models, running urban scale in operate mode
    
    * Add operate example model to tests. TODO: more results tests

commit 1bbea489d788e8cb06fd231c25b928338d204c44
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 19 09:09:36 2018 +0100

    Update example model colors

commit 37578a90adcb328cf80bb03b6a9f8021210a1451
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Sun Mar 18 16:22:20 2018 +0100

    Update docs; minor fixes
    
    * Add math to docstrings of MILP, export, conversion, conversion_plus constraints
    
    * Fix allowed constraints to account for constraints not triggered for certain tech groups
    
    * update visual representation of tech groups

commit 731af752249dc272d34eec3dd52f9be36197f515
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Sat Mar 17 22:52:23 2018 +0100

    Update docs
    
    * Add new conversion_plus section in advanced functionality
    * Add info on 'get_formatted_array'

commit a7fc9a32be509f5144606609c3347990c9a2d353
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat Mar 17 14:07:49 2018 +0100

    Fix plotting tests

commit 92c9b40f4a3ac8fba27ee1af566f7f56f80843b2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat Mar 17 11:47:40 2018 +0100

    Docs: fix math rendering

commit ab372ee47dbfaaae5d3c97b81e9a61eaae10dca3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat Mar 17 11:47:20 2018 +0100

    Reorganise plotting funcs

commit d1ab82f55242e70a47f951ff6b8869c3f42189bd
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Sat Mar 17 11:35:12 2018 +0100

    Move Julia code to CalliopeBackend repo

commit f01a5cca2ae8e77c687e8b77eb08e9ab157fdce6
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Mar 15 23:16:46 2018 +0100

    Fix operate mode; update running backend

commit 2f121455622bec6fdfc26816a63f75a18787cc80
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 15 17:56:41 2018 +0100

    Doc updates

commit 1c79c7d7a4d740dca49314d557e06eadfcc03dab
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 15 16:39:31 2018 +0100

    Permit plotting models with no transmission

commit 8341aecd4296c94452cd78629e342d923ea2bdad
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 15 16:32:59 2018 +0100

    Doc updates

commit 833cef9567d63aea6d553d32c0ee7c128f09aa00
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 15 16:32:39 2018 +0100

    Permit removal of model components

commit ecc3d9ba3a241b7119b788371ac9742f43763b88
Merge: 30cabc6 7a212f9
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Mar 15 14:50:44 2018 +0100

    Merge branch 'master' of https://github.com/calliope-project/calliope

commit 30cabc6e1463fd18d1f0f5c0d8e0da5337c3795d
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Mar 15 12:38:17 2018 +0100

    Fix resource_con variable gen condition

commit 7a212f93ed06555f6c78c23e698b3397df299df6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 14 17:09:24 2018 +0100

    Allow overriding built-in tech groups

commit f1e0b0fc0244c1a9462fd814ed562ac98be23f83
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 14 17:07:53 2018 +0100

    cli: fix debug and pdb args

commit 5c93526da9b99b32f319344a28cd4d306c8766b2
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Mar 14 15:15:38 2018 +0100

    Move configs in yamls; backend interface funcs; new params in model_data

commit d4cd696ff56213cd6403e96e35e04dd363952653
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 14 14:51:53 2018 +0100

    Docs: better tech group table formatting

commit 9f1d7cae0c3fbb065855dcbce364b58e32f178b4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 14 14:44:05 2018 +0100

    Doc updates

commit 8febe2265ff7ce5b1a65ba40f68c382cd5116bea
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 14 14:43:15 2018 +0100

    Save HTML plots in scripts

commit 14eda9d3e54d1cb05c0c17965fffa59fa24d3dbd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 14 14:22:51 2018 +0100

    Update model config layout

commit f175de3b3cb6416622761821bc61b57395b5ec7d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 14 09:40:37 2018 +0100

    Fix solver_options passthrough

commit 685a4a7cb8b0999ed7398a8fda34569f2c54bd20
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 13 23:52:26 2018 +0100

    Fix operational mode preprocessing checks

commit eb0cd3909c47a02ebb2e8dd8fbac1015e16dbea5
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 13 23:11:51 2018 +0100

    Fix error in time slicing

commit 9f7cfd0726666bd5041d14f89432d5c0f7a3ac09
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 13 22:51:45 2018 +0100

    Fix tech subsetting in plotting

commit f67563a52ff94ee2b3e02f197e42c80bdcf430e9
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 13 21:35:58 2018 +0100

    Update docs; minor fixes
    
    * urban scale transmission plots have a more accurate auto-zoom
    * Overrides can change the coordinate system (x/y to lat/lon)

commit 1354d1c972b5f3425f755776fa43b79ebd93046e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 13 16:54:02 2018 +0100

    Docs: add plot examples

commit 84942564800b2c52095a4bdea97c7f03df2d9edb
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 13 16:52:39 2018 +0100

    Add capacity factor and levelised cost to results

commit 4c11681fb3d0deba2fb8a938577ff29fe7afb5df
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 13 16:40:57 2018 +0100

    fix plotting, update docs

commit 8bdce5b7022a11b0471592226efb70d1f0b17614
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 13 16:24:45 2018 +0100

    Update plotting; remove unmet_demand; ad tests
    
    * Unmet demand is now a decision variable, only triggered when
    you 'ensure_feasibility' in the model-wide settings

commit 427bad60ef6430e9384b81a235efe2ee86aaf241
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 13 12:25:47 2018 +0100

    YAML conversion update

commit 34bbc56ea460fe70a13cf82eb95a0ac5741c6c3e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 13 12:25:23 2018 +0100

    2018's late arrival

commit 561f901ec1a099594a4b8fd759a354f09f0eac08
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 13 12:23:50 2018 +0100

    More doc updates

commit b9112a692c0c535a76cdac19ce26a2d34e0b1611
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 13 11:59:19 2018 +0100

    Doc updates

commit c9113d9412824e83a0bca2e11d579859c5b551e5
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 13 08:51:11 2018 +0100

    Update docs

commit 3d80490a2661769676b1394794e2349a75e62a41
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 13 08:17:27 2018 +0100

    Update requirements

commit c8962bd4c1578c10e3fd141c94176423d44827d7
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Mar 12 19:31:15 2018 +0100

    Update docs

commit a4897fc2e610b225354dc8b47db9e56d47a0b107
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 12 18:57:00 2018 +0100

    Clean up Model docstrings

commit ec9efb35e76a14be1f19e2b8adb77df17a5b49cc
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Mar 12 18:36:56 2018 +0100

    Update tests to include exact urban scale values

commit 6e8189f53338afdef66f5696800916f107eac4e8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 12 18:18:32 2018 +0100

    Minor plotting improvements

commit 9f5af28946a6e78f083cf4824bbcc6854f3ad564
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Mar 12 17:44:20 2018 +0100

    update defaults; fix operate safety switch; add to docs

commit 94072a3b34886677dd064346fbefd193b6bb7db3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 12 17:38:12 2018 +0100

    Remove stack_weight traces

commit c8f84151e3bcccb3c3d0e70e4b38f3a84d53dd28
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 12 17:34:02 2018 +0100

    plot_summary docstring

commit 53a0ba1765869feac2f1e3e470ee63b6af86125d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 12 17:28:58 2018 +0100

    Add single-file HTML plotting

commit 6106a6fef16762267067ee7648b6c34f6684d51d
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Mar 12 14:35:51 2018 +0100

    Update plotting to include dropdowns

commit be7202501fb7880423e30e5556c84b34265137dc
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 12 14:30:33 2018 +0100

    Docs: sidebar shortening

commit 9ee522228efeec025a99c18c0ffcd9175729abe9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 12 14:14:00 2018 +0100

    Doc updates

commit 4b78a6d45ea1f924127fcaef4a993d2bd8f0c1d9
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Mar 9 21:15:43 2018 +0100

    Add tests; update html plot output
    
    * Added charge_rate to csp in national scale model (otherwise it doesn't pass a test)
    * Updated constraint_sets so they build more correctly
    * checked constraint_sets.yaml to make sure they're valid for the example models

commit 20719665757fc897802f24965538e234d3419ffa
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Mar 9 17:50:20 2018 +0100

    Allow html plotly output; add tests

commit c5a62eb34c1487737aecdb3c6457e97af40dfa51
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 9 17:20:44 2018 +0100

    Fix time resampling, test it

commit 448b5c1cc75c7094794a3592d8d73b9878d13bc9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 9 14:54:45 2018 +0100

    More plotting improvements

commit b0c9820c8cc4a4819dc17040a527665755cd8ab7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 9 14:38:58 2018 +0100

    Plotting improvements

commit e71cf026b8f79b3571637ec89c19765712991332
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 9 13:10:12 2018 +0100

    Add help for contributors

commit 24dca7ade502e2b38095c8fef691ad20b8ea7aa6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 9 12:14:53 2018 +0100

    Docs: recommend miniconda

commit cc893f46f733ef265d99deda5de5404f9fbe7a3f
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Mar 9 11:01:04 2018 +0100

    Added tests; updated docstring equations

commit 5cfab0c4c81ebeb477e905f9cf165be9d63fe95f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 9 09:42:03 2018 +0100

    Fix RTD badge

commit 717c499219fef293d9733642481fb90171b87e8d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 9 09:40:39 2018 +0100

    Revised README

commit 0ee689c90085adfd55b9fc73ed2935a71a0c8cb2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 8 16:44:02 2018 +0100

    Clean up plotting

commit 7a5c2946e62ccf6955a2575c2e7ab4086e0d3d24
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 8 15:02:39 2018 +0100

    More doc cleanup

commit ab88fb69a5ed4ac9b4de58e65a850d020e80ff6e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 8 14:54:47 2018 +0100

    Fix docs sidebar

commit ab6836f4354a14902ed718d560cf976fa3ef9a17
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Mar 8 14:31:51 2018 +0100

    Update documentation
    
    * Begin moving constraint formulation to function docstrings
    
    * move reserve_margin constraint to policy.py

commit 74bbeafec70e510d7a081788a5d2f80c4bab01ac
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 8 14:17:23 2018 +0100

    Doc updates

commit 7ae9007d0e912983662072ffd0d57001f79c0c3a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 8 10:06:30 2018 +0100

    Fix warnings

commit 86d816d332cab48cc3b466e3091a7456405008de
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 8 09:58:49 2018 +0100

    Warn on unboundedness or infeasibility; other minor cleanup

commit 7d00f4fdaffb2ba4d29992d66ce2df851751b9d8
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Mar 7 20:26:45 2018 +0100

    Multiple fixes; added tests
    
    * Updated sets to avoid duplication of milp technologies defining `purchase`
    * Updated timeseries dataarray generation, to not lose information on static values
    * added tests, updated checks accordingly

commit 565a5ee60a80bdeb54a253b2034baf3130ddd614
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 7 17:14:59 2018 +0100

    Docs structural changes

commit c71cea5b30f52851d7290cf58334aa1e7fb04464
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 7 16:20:10 2018 +0100

    Add cbc to solver tests; pyomo from pip

commit 33dbdc1e26c624209e76addd048c852bd1a32221
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 7 16:13:12 2018 +0100

    Permit models without links

commit 212182eb010b6bf98d9cdbc56ea9f34f53ffe235
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 7 10:45:00 2018 +0100

    Add cplex and gurobi tests; bump pyomo and numpy versions

commit 66ecc66b092a58ea823b24f18e54ee8f8e85ad4a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 7 09:05:22 2018 +0100

    Bump pandas version

commit b3ce3ea9f5b4c7bfd357b1850265a7b16e47fa0e
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Mar 6 18:29:53 2018 +0100

    Added tests; updated checks
    
    * `resource` removed from supply abstract tech group required constraints

commit 6ee8cd6054ffb7d6c4a5670e1c217a00443e7e3d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 6 17:21:54 2018 +0100

    Add version checking

commit aa0bbf487896d10078a7b56779fd55ba58a48df5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 6 16:31:07 2018 +0100

    Re-include 0.5.x changelog

commit e399f5c523628bcd894e766524f4e621bb6f8a9e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 6 11:19:41 2018 +0100

    Fixes to docs

commit be5e8c38132cb696273ac0b456828fd040d3da7d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 6 11:19:21 2018 +0100

    Run generator improvements

commit 8977b85023adcbe10ac99a678e435283ec709425
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Mar 5 22:18:59 2018 +0100

    Update docs; add tests; fix policy.py
    
    * Updated file pointers in documentation
    * Added preprocessing tests and fixed code where tests failed
    * Added `tech_order` argument to plotting functions, to replace `stack_weight`
    * Ensured group_share constraints are actually added to Pyomo backend

commit 3b964be96f99d9c0783b85f0060d7420a1fd616a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 5 17:55:25 2018 +0100

    Run generator re-added

commit 1e0130ee2991916393f7ed081e9f614152063382
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 5 13:34:44 2018 +0100

    Further doc updates and fixes

commit 2cd7a8938c6aa172063deb38a5d2e0ba7df32320
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Mar 5 12:15:04 2018 +0100

    Updated link tech definition; added updates to docs
    
    * links now define transmission technologies under the key 'techs'
    
    * `updates.rst` added to documentation, to detail 0.5 -> 0.6 updates

commit 178703ebbade02cb6f16360f5cc4b67e320860ca
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Mar 1 10:44:33 2018 +0100

    Update docs; Add test model; Update checks
    
    * First few pages of documentation updated (build to HTML not tested)
    * Basic Calliope model added in test folder, for use by all tests
    * Added checks, inc. removal of empty dims from model_data
    * Updated electrical energy carrier to 'electricity' in urban scale example

commit c07634ddeb6f7a7fa5b563c9187700110bb04f35
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Feb 28 12:22:04 2018 +0100

    Fix updated xarray indexing
    
    * xarray 0.10.1 created a lookup array of numpy 1d arrays, not of strings (as it did in 0.10.0)

commit c50a3b67ee661a937d804eb03f91e1d2865f9e3e
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Feb 28 11:08:00 2018 +0100

    Add operational mode
    
    * Turns capacity decision variables to fixed parameters
    * Added checks for consistency of input data for working in operational mode
    * Converted timestep_weights and timestep_resolution to Pyomo Params in backend
    * Added Plotly to setup.py

commit 08eddc90080784600b8fd44c35928263a490fb91
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Feb 22 16:36:46 2018 +0000

    Update MILP constraints; other fixes
    
    * Working MILP constraints
    * Fix which sets 'resource_' constraints are assigned to
    * Added units_capacity to MILP constraints
    * Allow energy_cap_per_unit as an option to meet required constraints

commit ed6860700eb977c2a10c03c94cd75cfebd149753
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Feb 20 12:30:42 2018 +0000

    Minor Fixes
    
    * Plotting tranmission ignores unused technologies
    
    * preprocessing correctly captures per-link overrides
    
    * Preprocessing captures one-way transmission and Pyomo tranmission balance constraint picks up the change in `energy_prod`

commit 750852468f7c6051ded9d79ab48881812c82efd9
Merge: 889edf7 3eb1308
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Feb 13 12:23:50 2018 +0000

    Convert Pyomo data dictionary to Pyomo Params
    
    Feature can be used to dynamically update parameters without rebuilding
    the Pyomo model.

commit 889edf7ba657d666d1c2ae80c3802ba476be53fd
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Jan 31 14:56:39 2018 +0000

    Update README; Minor fixes
    
    Minor fixes:
    * Add 'resource_min_use' to defaults
    * Add **solver_kwargs to Pyomo model solver, to allow e.g. warmstart
    * Change all .to_pandas().index to to_index() for xarray DataArrays
    * Change '_timesteps_per_day' (single value) to '_daily_timesteps' (list)
    to allow different time resolutions in a day, if those variations are true
    for all days
    * Ensure timesteps are the last dimension on stacking dimensions in time
    clustering

commit 41002edbee5221d31eae0715a675467c51b95b79
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 31 14:02:32 2018 +0100

    convert: add default interest_rate and lifetime

commit dc1bd15e38ef3bd60c55e80f5a8c32848e8169e5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 31 13:08:27 2018 +0100

    Add group_share constraint, unified load_constraints func

commit be7bc8b8afbe8eacba31093a0e2708d9b3a621fe
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 24 11:45:05 2018 +0100

    Fix inclusion of conversion YAML in package

commit ed5295c3f30672135c4e3c303b266fd78da519a3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 24 11:28:12 2018 +0100

    Initial implementation of 0.5-to-0.6 converter

commit 8ea5be92cae44b77d92aed3612439f6e28227772
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Jan 18 15:57:39 2018 +0000

    Improve constraints; Update timeseries resolution
    
    * Conversion plus constraint error removed
    * Export cost constraint error  removed
    * Timestep resolution given for every timestep in series, not just based on the difference between first and second timestep in the series

commit 9b92a65c793a433a1abcbcec46731a025e158f3d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jan 18 13:16:59 2018 +0100

    Add constraints: reserve_margin, resource_min_use

commit 907e46b263db6e1e4d6b00b723bb84f529678a3e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 17 15:08:43 2018 +0100

    Add more constraints and clean up aux files
    
    Adds:
    * energy_capacity_systemwide
    * energy_ramping

commit aff217829aed3ee29ad1bf7285aa24a4f9becfff
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Jan 8 16:20:09 2018 +0000

    Update conversion_plus constraints; add checks
    
    * conversion_plus now uses the correct carrier_ratio in in_2, in_3, out_2, out_3 constraints
    * conversion_plus now catches all loc_techs with carrier tiers which are not 'out' and 'in'
    * Added model_data check for resource = inf when force_resource = True
    * Added model_data check for cap_cost = negative when cap constraint not set (for all capacity constraints)

commit 3eb130866e4d6438d10f25a0f1a454202c047a6f
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Jan 4 15:43:19 2018 +0000

    Make all parameters Params in pyomo backend

commit c85a372ac82d80b685bd6f89ffeb3819b0fb1b5e
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Jan 4 15:41:18 2018 +0000

    Minor fixes

commit 78cdf33ed52919a1b95a6cae514cae0de350156c
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Dec 19 10:06:37 2017 +0000

    Update Julia backend & time clustering
    
    * National scale example working in Julia
    * Urban scale solving to incorrect solution
    * Defaults stored as YAML string
    * Run KMeans from scikitlearn
    * Automatically get number of KMeans clusters using Hartigan's rule
    * Fixed some Pyomo backend constraints (in MILP and storage)

commit 259f9ddf80d7f951028012c7195d7af55302be2f
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Dec 18 09:30:46 2017 +0000

    Fix constraint sets
    
    * National scale tests work again
    * Urban scale model still not acheiving 0.5.x output
    * Added constraint doc-strings

commit 3e16bad42ee9cfb1caac29bffbc6526da165277b
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Dec 15 21:15:43 2017 +0000

    Add constraints and constraint_sets
    
    * Add working export and MILP constraints
    * Disable total costs test in National Scale test (~100 lower than expected)
    * Create a loc_tech/loc_carrier/loc_tech_carrier set for each constraint
    * Move MILP and export constraints to dedicated files
    * Update transmission plotting, with intellegent mapbox zoom level

commit ca95235c94c9968e8b4a7bcf01c08aeaa6648db8
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Dec 13 21:39:06 2017 +0000

    Add constraints and restructure files
    
    * Added conversion plus, conversion, and export constraints
    * conversion_plus, conversion, and export constraints are in their own files, with Pyomo Expressions used where necessary
    * Added per location available_area constraint to capacity.py
    * Broken out data.py in preprocessing to model_data.py, lookup.py, and time.py
    * Increased number of lookup arrays in model_data
    * working urban_scale model (but with different results to 0.5.x)

commit e7250392d824837636c4e97f21ac55f7e1521756
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 13 10:41:42 2017 +0100

    Minor cleanup and preparation for more constraints

commit 1130920e545cbfc7b4f727daab48c554c8809f30
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Dec 11 23:16:26 2017 +0000

    Add basic plotting functionality
    
    * Model.plot() can be used to call plot_timeseries, plot_capacity, and plot_transmission
    * sets.techs now includes transmission technologies without their links
    * technology names from essentials added to model_data

commit 84724da70cb9cd14c26b7dd5f094d713f3193897
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 19:31:42 2017 +0000

    Implement save_csv and save_netcdf in cli

commit 45ffa9e98759392ae36ad77bd848f70cf7fc1f28
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 18:27:18 2017 +0000

    Remove old analysis code and matplotlib dependency

commit ea0d1e3db989c015e9aa7d8310a181c6e3c53c3e
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Dec 8 17:48:31 2017 +0000

    Add force_rerun check to Model.run()

commit ad454a323f4eb6df7974aa78b3c3b748129d44c1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 17:39:19 2017 +0000

    Make is_result attr int rather than bool

commit 4813b42460c4fa5d9644f10fed3f77781cebf0e0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 17:35:20 2017 +0000

    Implement read_netcdf and to_netcdf

commit 819c8d4993364a72bb5fa160b0f33ccf42ea8457
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Dec 8 17:33:15 2017 +0000

    Update split_loc_tech; fix to_csv folder path

commit 00bd04fd574357e52f8f05febd507eb4e7873e0d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 17:19:53 2017 +0000

    Barebones input/output

commit cf099fa2ef7e60176c77aa66e21f9f708161b44a
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Dec 8 16:26:21 2017 +0000

    Fix split_loc_techs & add function to Model object

commit 2d8f7b1d0992212dff05b157478fb9f283a3787b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 16:17:25 2017 +0000

    Merge results with input data in core

commit eed02ee165f2b9629f73f8b5610c9e3b19910046
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 15:19:57 2017 +0000

    Add sense-checking of national-scale example model

commit a7cb875510ae11a649d4ecdbe323210a5d40f98e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 14:03:48 2017 +0000

    Fix vincenty import

commit 13de4b490653e8224f10832860d6992bd0df842d
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Dec 8 12:28:41 2017 +0000

    Clean bugs in change of loc_tech separator

commit 23420966f34ca7f53dcfbbfa0357507f7cf4fb4c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 12:16:24 2017 +0000

    Rename param_getter to get_param

commit 2194d66ba57433b1dfea0db33c6c122e2811da49
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 11:34:22 2017 +0000

    Fixes and more tests

commit 9e2ab98b1aeb82b5834f699c22f934da9e0385b6
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Dec 8 11:26:20 2017 +0000

    Reorganise util files; update loc_tech separator to '::'

commit 4c055d095460ef32a290cfcac051b3039b6632ca
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 11:15:25 2017 +0000

    Fix variable cost constraint

commit b066e05645c3e72522cfbe18e2338f284ec2d0f0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 8 10:15:25 2017 +0000

    Fixes
    
    * fix param_getter
    * disallow parasitic_eff for storage

commit 14aff177126fe206c047c7388d416d12c40ff44a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 7 20:20:19 2017 +0000

    Fix get_timestep_weight; fix national-scale interests

commit 532ebe42b6d10bc2f131d391e10c1f59c6c05866
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Dec 7 20:16:32 2017 +0000

    remove s_time_max in national scale example

commit 5a791e1620c62fd2a1100f137bd8b1ad4337f369
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 7 19:11:20 2017 +0000

    Reorganise dataset dimensions on model results

commit b1aafc91644e4754bca8cc4be4a35cbf19c52e69
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Dec 7 19:08:59 2017 +0000

    Update constraints and other fixes
    
    * Add network constraint
    * Add capacity multiplication to investment costs
    * Remove unnecessary package imports in constraint files
    * inf to .inf in defaults.yaml
    * remove `resource_scale_to_peak` and scale demand to old peak in the csv files

commit 23fef7ad7a798eb7826d67424949b59a7a39e748
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 7 18:50:19 2017 +0000

    Return raw model results as xarray Dataset

commit 95182abe0d63061667ca6c1e46f169347ae46f53
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 7 17:49:46 2017 +0000

    Fixes
    
    * urban-scale example
    * location preprocessing
    * downgrade pyomo to 5.2 for windows compat

commit a05a931ea2f523ec7ae07a33bd6e86f3621f86fc
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 7 17:02:51 2017 +0000

    Add variable costs

commit c34089958a75b8c2bb5dc1860d5fcd784d02a4c9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 7 16:23:16 2017 +0000

    Add some cost constraints; various fixes
    
    * Compute depreciation_rate in preprocessing
    * Fix objective, energy_balance constraints
    * Fix national-scale example model

commit ecfaea2998fba50dfc61def166036047945cd6f6
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Dec 7 16:16:54 2017 +0000

    Add dispatch constraints; update inf to .inf in defaults

commit 1742852e146b537caa21bdf610367f7fd26cca73
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 7 14:06:51 2017 +0000

    Fixes
    
    * Fix get_loc_tech_carriers
    * Fix set names
    * Logging improvements

commit 10a82bb53550ab51750e9c79fad8464c46786be3
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Dec 7 12:39:22 2017 +0000

    Add more Pyomo constraints; update Julia constraints; add loc_carrier lookup

commit dcab9137e95349d00d88118ee0b1d4cb3036f27b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 7 12:21:27 2017 +0000

    Add system_balance constraint; additional sets

commit 2398005fdfc66a0765f4494cd72d3c125de270ed
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 7 10:32:53 2017 +0000

    Add basic logging

commit e37ca456840db2c7082e6a40e4ef524fbcfff2c3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 7 08:33:32 2017 +0000

    Initial working version of pyomo constraints

commit e21a40d89129a7c92910e7c34b0021dbe02daf80
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 6 20:11:06 2017 +0000

    Fix pyomo req

commit e27c43007651b792c3fe0cb89519ce5510577e9d
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Dec 6 19:37:42 2017 +0000

    update Julia backend; add lookup_remotes

commit 686127a9835db19aecc5215f9d4e179c6bca13a0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 6 18:55:36 2017 +0000

    Improve serialization of defaults

commit 96fd30835cf902163abbde77a6c08f5e7cc1a380
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 6 18:40:42 2017 +0000

    Serialize defaults in a model dataset attribute

commit a9e07a8c7e784c36527af08b6cd0b41fc1bd261f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 6 16:53:36 2017 +0000

    Add barebones pyomo backend and run method

commit f9ad3d38014b3fb82ecda52570f968a91737066c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 6 16:46:59 2017 +0000

    Fix per-distance computations

commit 220e4f74a5db8ec520bf12d2b093803c4c1bc7b9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 6 16:25:41 2017 +0000

    Permit args to pass through to example models

commit 0472e4fe6c93a24f66861574ea94217062bdf88f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 6 16:20:52 2017 +0000

    Compute per-distance constraints in preprocessing

commit 670dd8522830f2e43c0349d5bd59439b9a1c92bc
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Dec 6 16:11:13 2017 +0000

    Update Julia backend; add carrier lookups

commit 59b4d4e1288c56b29bea6f429e1ae303722d6ee8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 6 15:22:49 2017 +0000

    Additional sets

commit 4926943d505a3b71c84a31e74be7ea91fa437be1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 6 10:18:52 2017 +0000

    Permit multiple override groups; minor cleanup

commit 073961fbc046b1ca877b7939ad115e9f3ea1fd01
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 6 10:13:27 2017 +0000

    Update requirements

commit 28b223390d452c7716a7cdb9042d2189664feb20
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Dec 6 09:51:14 2017 +0000

    Add Julia backend

commit a44466b7f45b82bf55e0b91d526ce7bd653c23db
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 5 15:21:45 2017 +0000

    Working CLI; fixed Makefiles; minor cleanup

commit a5f252637a3d779bb722a7078aba340c8f491e63
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Dec 5 15:14:36 2017 +0000

    Add initial tests

commit 2322caff2ee9db43f870d6af26e3e627d9236bfc
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 5 13:13:41 2017 +0000

    Fixes to time clustering

commit 25b71f5c13a275d95b451faa4ba7846a94c9170e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 5 11:49:08 2017 +0000

    Further restructuring and cleanup of core

commit 06262b75ef130d743302ca18c4ebe9a4463cd6df
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Dec 5 10:57:43 2017 +0000

    Minor fixes

commit 151462a44280ed0191b2e67f1bf6527b4ba7821f
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Dec 4 15:23:42 2017 +0000

    Read timeseries data from model_run

commit b1f70b98eb1e94e64ffecbdc5d50f4d7fec46dff
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 4 14:37:57 2017 +0000

    Read timeseries and build timesteps set for model_run

commit 9d5c390b7197d5b3c2c94234295ce2c130cff5ca
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Dec 4 12:31:17 2017 +0000

    Update model_data time dimension preprocessing

commit 2ea865754b202ea4a19a7f5960fcd7b8481a1f6a
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Nov 29 16:39:01 2017 +0000

    Ensure static parameters in resampled timeseries are caught in constraint generation
    
    Fixes issue #80 where a static available resource, following timestep resampling, was not being correctly updated in constraint generation. Increases information held by timeseries Pyomo Params, which will increase Pyomo memory footprint.

commit 21eb337586e99f4bc07bbfd131a782537806ab60
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Nov 27 13:18:39 2017 +0000

    Update time clustering, resampling, and masking

commit a974350eef0795af7498a1c9c24a9754bea87d67
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 24 14:23:49 2017 +0100

    Additions and fixes to checks and preprocessing

commit 1eff344a0a707466c5f1b430630727713746bf66
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 23 17:06:49 2017 +0100

    Further preprocessing checks

commit dbd9f216ad35fb1a2913a99360e423e39081ef98
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Nov 21 11:43:04 2017 +0000

    Fix inputs required to set r_area to zero

commit 93e18977a95e5bdb6a2bdc0f02bae0284b0aa7e7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 10 15:59:44 2017 +0100

    Release v0.5.4

commit 2cb51c7d665d23d074e995063865939454b7e06f
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Nov 10 12:07:39 2017 +0000

    Remove necessary links and techs on using subset_x

commit 0c5021cb9c4b0d941b808fd08c5913a6012f346a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 8 14:37:00 2017 +0100

    Fix stack_weight datatype on reading NetCDF solution

commit 1131ae3f72533e57b526a46fb392e10fdc787665
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Oct 30 19:30:45 2017 +0000

    Fix ignored r_area.max & r_cap.max; allow plotting carriers w/o demand; update changelog

commit 9862a2fc2b726cf068708b70fb2594e8ceebe268
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Oct 23 07:44:56 2017 +0200

    Update preprocessing checks

commit 2b338d312ba6fa850d6e05a16b102f89fe55b091
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Oct 19 19:37:43 2017 +0100

    Add `carrier_tier` preprocess check; Minor fixes

commit a28a76824a86319353a0e3e372be86a007dbb51e
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Oct 18 15:39:03 2017 +0100

    build model_data from dicts and loc_techs subsets; update set generation

commit ca2429d435764010a5c720c85cc7fb1b743fe657
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Oct 17 09:17:49 2017 +0100

    Update essentials_to_dataset

commit a9f7aa66b9c594ca61e961865aa7516912ef3857
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Oct 13 18:53:59 2017 +0200

    Add more debug comments; model_run.sets to lists

commit 10f6c0f199ce559394747b34846f251431547c0b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat Oct 14 12:29:29 2017 +0200

    Sets: transmission included in loc_techs

commit 2edc8abae048703938f079a7a9bb23d6a6dcfc90
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 13 17:41:32 2017 +0200

    Improvements to preprocess_data

commit 89895d9d7253a3fdf740b4aa4d7bfb7e587e0644
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 13 12:00:40 2017 +0200

    Fix misplaced function name

commit 1dcf7c6ca370f9eb38cc83f9197982a7118222d2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 13 11:58:12 2017 +0200

    Pre-processing consistency with data generation
    
    * model_run generation changed
    * Add tech_groups to model_run
    * Set names now plural
    * Add some docstrings

commit 6d7ef14431a32ff4cd23565893163010611f1c94
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Oct 13 11:55:13 2017 +0200

    Update Model initialisation & model_data build

commit dfe23b6a8a4c962f9fd9781b97dba38bf0a012ce
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Oct 13 11:18:54 2017 +0200

    Add model_data preprocessing functions

commit b259946f0a8e40eba4b1a89d60b058327702ef49
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 13 10:33:13 2017 +0200

    Resolve column names for CSV files; fixes to names

commit f4984f1d4cc844e56e8c5a52ea09f56f6a4b4b7c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Oct 12 16:11:12 2017 +0200

    Rewrite model preprocessing in core

commit 48fe4749632b77957379ddd2c7175e9c8ea0254e
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Oct 12 10:03:12 2017 +0200

    Fix storage and time masking issues; update MILP example

commit 153d4d8e5b86005b29bb5dd9853384aa36ce74de
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 6 10:27:37 2017 +0100

    Update to ruamel.yaml; change order of relative_path arguments

commit 2a06a2f4ac51cbf4a8ff9e3f6ea567dafd38f677
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Sep 11 16:22:36 2017 +0100

    Merge technology and location sets

commit 77a8cee35bff4b0440a05510fc0b0ce91537bc3f
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Sat Sep 9 16:00:40 2017 +0100

    Ignore unmet demand e_cap in operational mode.
    Fixes issue #68

commit aea97391f786e5986e6048942ec142451d634aeb
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Aug 22 19:17:01 2017 +0100

    Release v0.5.3

commit 0ac977b13ec161f366d9a523b9c5837fbc28846a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Aug 22 17:31:04 2017 +0200

    Add time-varying costs to solution

commit 91403b09821939a99ef73f1b5792ad9c9b61e57b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat Aug 19 10:24:11 2017 +0200

    Non-destructively save solution to netcdf
    
    Closes #62

commit 2925dca8ad1fe6bd37078bd65113f6bab94b3c93
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Aug 14 18:23:27 2017 +0100

    Fix some time clustering bugs (issue #66)

commit 77a09302699db1f43806949cd9f05dc1d754c74b
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Aug 3 17:37:26 2017 +0100

    Fixed issues #61, #63 & #64

commit 2a99495abc770a9038b0f2de3d22cb9e70a9cfa2
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Aug 3 12:50:04 2017 +0100

    Fix one way transmission

commit cc9536c1156fdb7e296272dbbed153f2524d808d
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Aug 2 16:14:25 2017 +0100

    Updated changelog; Fixed issue #60

commit f0d5ccf7b8c833e7ccd5588a96b941fd0bf796e0
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Jul 18 14:07:04 2017 +0100

    Updated documentation; MILP fixes

commit 805e60afd150a821fe128d634b64a9ac190db133
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jul 11 15:00:06 2017 +0200

    Documentation and README update

commit 4ee6d7cdbe5c86328fc9c7c1b2e5fa5a609a591e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jul 10 10:11:52 2017 +0200

    Switch from pyyaml to ruamel_yaml

commit 0a0c1c0b05a9ffacbecd90c903e67aebe99a2eea
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Jun 21 16:37:04 2017 +0100

    Minor fixes incl. MILP, ramping & PV urban-scale tech

commit 6f660655188afc0833dccb3a78914584704ddf7f
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Jun 21 16:00:04 2017 +0100

    Adding milp functionality

commit eb4a9f8f727f9e02642ba8017bcd3f2c80bd61ec
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jun 21 08:36:02 2017 +0200

    Add microsecond to time-based run id

commit fb84f06428fc903211b36e8ee5925547c4a348c5
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Jun 20 16:08:08 2017 +0100

    Add binary variable linked to purchasing cost

commit 366d22a19d620b0fda0ada2c1845faac1f0b7a9e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jun 16 16:11:17 2017 +0200

    Release v0.5.2

commit 901dff43e1389aea8c4b5d2475605949e5794153
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jun 15 15:36:30 2017 +0200

    Downgrade pandas to 0.19

commit 2b4225487846e4f537ce2a22e33e0313fa063273
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jun 15 13:06:06 2017 +0200

    Multiple bugs fixed

commit c81b8c1f8434c0c2d4c811731f29ef460efd2350
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Jun 14 16:29:49 2017 +0100

    Update requirements; minor fixes

commit 788c50bca1f14d57191dd01a09c4550b7d9d1cbe
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jun 14 16:00:36 2017 +0200

    Fix py3.6 reqs in 957f008

commit 957f008164d8124af08848a1a3c275c97d223aa5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jun 14 15:35:53 2017 +0200

    Update requirements, move to Python 3.6

commit dce24d9ffc5b9c4b131f43a43af9dee21cba6f27
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jun 14 14:31:07 2017 +0200

    Release v0.5.1

commit 4fe3485940a193e55a8a3f69f5e3ac2bd7c7023f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jun 12 10:50:21 2017 +0200

    Add hound and flake8 config; minor cleanup
    
    * Fix Appveyor badge
    * Update .gitignore
    * Update development.rst

commit 35d2dfa60939db60272f64a93c5434767b869358
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jun 12 09:59:26 2017 +0200

    Do not lint on travis

commit e12d6c00eaab372d45ef2ad6d540e018753441b5
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Jun 9 11:34:07 2017 +0100

    Update max and min prod constraints

commit 99b7ea730d902f2289022eca9eb819a5edf5a585
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Jun 2 18:15:03 2017 +0100

    Update documentation

commit 81d75af6d438fc04f21c9845cae3051f4879de9c
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Jun 2 17:23:05 2017 +0100

    Add one way transmission; Fix issues #49 and #47

commit d78b0cb2cb8f24e95c5370906fd48f3be7c6dab1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jun 2 09:05:09 2017 +0200

    Fix setup.py: search for and include all package data
    
    Closes #45

commit 144b0d5b23953db5a9b29ff0a058a3164e193b15
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed May 31 10:52:28 2017 +0100

    Update YAML entries for location metadata

commit 8989ada32e19cad040c4000da6f03067fce8477a
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri May 12 12:23:57 2017 +0100

    Update map plotting for x-y and lat-lon

commit 3874280e5925e738d48f7b7d73931dff8e40e173
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 5 07:57:41 2017 +0200

    Add conda-forge to release checklist

commit be63d40ef2c400cd82729264867ce136ce01b77d
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu May 4 20:32:02 2017 +0530

    Release v0.5.0

commit 6f02d6670a360601243b38042749fd9fb530166e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 4 09:09:34 2017 +0200

    Update requirements

commit 2d7c6cce185dce251974bb2d67e6da02b310aa45
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 4 09:08:36 2017 +0200

    Minor updates and fixes
    
    * Add index to docs
    * Remove BaseModel class
    * Add development blurb to docs index

commit e0278bb8fd07b961fc5e3be66bc1b1bdaaad63b0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Apr 30 08:52:38 2017 +0200

    Switch to Cloudflare CDN for Mathjax

commit 52006cf54e918305f43cd5cad303db28993acccf
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 27 17:52:22 2017 +0200

    Force newer sphinx on RTD

commit 33a54d1a302b5cb2d87f74ac08f9e6924e1e019e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 27 17:33:35 2017 +0200

    Re-add ability to choose carrier in get_summary()

commit 1e9a409806711d05391e99491dfafaecb77b2c4a
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Apr 27 15:27:38 2017 +0530

    Update get_summary function to choose correct carriers

commit 7a4aa268b3a775b438c20310c12a7c27de4b8630
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 27 17:05:49 2017 +0200

    Update national-scale tutorial; other minor fixes and cleanup

commit 20addebed5420949cf477460801a8f4c95e75d0d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 27 07:51:10 2017 +0200

    Move examples to a submodule; further doc updates

commit 91e7a37c7cd77c04b6a68048e2730569f97dd226
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Apr 26 23:31:22 2017 +0530

    Specify allowed constraints for each abstract base technology in docs

commit f3974ccd85ae577c7b8c210bc212d653046f02fd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 26 18:09:58 2017 +0200

    Pin icu=56 in requirements to fix Travis builds

commit c5feea5c2631a6c49f1250f1b642bd8ea8e8704d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 26 17:48:40 2017 +0200

    Further additions and improvements throughout docs

commit e1f1c9245d5d4599a9314f9eda4ccb3a51b23e30
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Apr 26 15:37:25 2017 +0530

    Minor fixes

commit 6ac97b6f1acc2404f09177e3e54859872df714c8
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Apr 26 14:18:53 2017 +0530

    Added urban scale tutorial & updated tutorial docs

commit 1536073dded58af73fc88dd06c2739aedce7512e
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Apr 7 15:04:57 2017 +0100

    Updated formulation & configuration pages

commit 8300559bec91e6df9bad20a9f19a929df4668eac
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 26 09:51:39 2017 +0200

    Updates to README, requirements.yml, and imports
    
    * Add gitter badge
    * Prioritise conda-forge to avoid conflicts
    * Remove unused imports in analysis.py

commit 13171cb23f869fb2aeef45e1719a96de2a70dd9f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 6 17:22:00 2017 +0200

    Update contributor statements in docs

commit d6b01070d81bda217344842ae129e7211ca88413
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 6 17:20:45 2017 +0200

    Fix analysis functions to pass all tests

commit 7be5563a5c6d90923e4eb054d4f582099cb626ad
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue Apr 4 19:18:30 2017 +0100

    Signed AUTHORS file

commit 3d550feee3b9e7ca6a9eb251f248521e632da9d0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 4 16:47:39 2017 +0200

    Update analysis tools for changed tech definitions
    
    * Update analysis for supply_plus
    * Remove SolutionModel class and associated tests
    * Bump version to 0.5.0

commit 6f2e1b83f1fb776e10f8918e8557b6c7b442ed8e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 4 16:43:02 2017 +0200

    Update author statements and citation info

commit bd5cbc938e091592547c06d522f14d57fdfb995a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 31 11:42:50 2017 +0200

    Clean up merge; mark CLI tests as known-fail

commit 08382a75dbcd7f07851ba19c0046005ae50de875
Merge: 424a0c0 49ee231
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Thu Mar 30 17:20:26 2017 +0100

    Merge branch 'master' into new_technology_definitions

commit 424a0c0a81979de6f191be362391b7b7365a0021
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Mar 6 19:38:00 2017 +0000

    Update tests & export

commit 49ee231e831890329fb9b3fde3bc3a9fb35e0066
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 6 11:07:23 2017 +0100

    Fix Makefile from e0a1195: spaces to tabs

commit e0a119560f73b83a1017e5aff46e680fa5c31832
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 6 10:01:56 2017 +0100

    Dump profiling output into CI logs; fix dev requirements

commit c8d16d35a70b1129f779b6a17a0ad16e9eef238c
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Feb 22 19:12:00 2017 +0000

    Robust s_time -> s_cap conversion

commit 33b6fa6ad772ac0c4e3b1f48d73d29d723b5a4e4
Merge: 2a52ae9 062dcbd
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Feb 22 08:58:38 2017 +0000

    Merge pull request #35 from calliope-project/revenue_as_negative_cost
    
    Define revenue as a negative cost

commit 2a52ae9cf50f874a9127cba346a078393f05e705
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Feb 16 17:19:14 2017 +0000

    max_r_area_per_loc optional constraint added

commit 062dcbd14f0d8b41809d2a24b06c7a21dadd286b
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Feb 15 14:05:14 2017 +0000

    Define revenue as a negative cost

commit 4a038914c26469494cfe711f91fde3a0a0093ef3
Merge: dab5b71 3e6de8a
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue Feb 14 12:15:33 2017 +0000

    Merge pull request #31 from brynpickering/export_patch
    
    Make exporting energy simpler to include

commit 3e6de8a9d32039847ea676ef37e61dfea30a7a21
Merge: be99be0 dab5b71
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue Feb 14 11:58:21 2017 +0000

    Merge branch 'master' into export_patch

commit dab5b71ed388314e682a1fbfba3222895ff9c446
Merge: a45c3e7 17ad3a9
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue Feb 14 11:53:26 2017 +0000

    Merge pull request #32 from calliope-project/per_distance_patch
    
    Ensuring per_distance constraints and costs are actually accounted for in the model, rather than the current state where they are ignored without raising an error.

commit ef2d0d279d8a2cd3b37db8c222241bb07394f600
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Feb 14 09:34:00 2017 +0000

    Generate new sets in Pyomo model; clean base.py

commit a45c3e79ccc3f40e20a93b8798293b06552b646b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Feb 13 17:06:30 2017 +0100

    Point CI badges at master branch

commit 7adeb25fc8e19278596908aba43ee699710d8dfe
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Feb 10 17:19:00 2017 +0000

    Fix functionality switch

commit 17ad3a97377cfb4fa065bf151d4273527a3ee39a
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Fri Feb 10 15:59:50 2017 +0000

    Include per_distance constraints in defaults

commit cd67035dc484f85f640c3ec20e3e6ae37efb65c1
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Feb 8 17:32:30 2017 +0000

    Minor fix

commit c9e0ca1fa1f49b76c7b11c355bedbd09297ef41c
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Feb 8 17:11:27 2017 +0000

    update per_distance functions to work properly

commit be99be0eabd2494eefe6c18c0da573e9123ae2b2
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Feb 2 17:07:29 2017 +0000

    Make exporting energy simpler to include

commit beefde1b68d414f7e06dc18d060e0c8df93dc0f1
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Jan 11 16:30:55 2017 +0000

    Optionalise revenue variable & constraint gen

commit 3cc41b3ab86c537555c529b5826f0301b6a973d8
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Wed Jan 11 11:22:40 2017 +0000

    remove get_e_eff

commit fdf8f2fc75372ae3064cb640129767c64cbcab2f
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Tue Jan 10 11:42:59 2017 +0000

    Add timeseries costs/revenue to param updater

commit 76b05bf9a6414a4bb64ddf2b51110f4ac91e8b6d
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Jan 5 23:26:51 2017 +0000

    Add empty step array exception in iterative solver

commit cbe0db77019d05c662a9da87a757a7b159b2d6b3
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Jan 5 23:19:04 2017 +0000

    Minor fixes

commit 40ed7bdd5b02d0954f628fd073a29385b3338d31
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Mon Dec 12 11:44:19 2016 +0000

    Fix error with validating consistency.
    Fixes model breaking when files are loaded with more locations than those defined for the model run

commit 1365ce08f6c5d38c150677a80cfbfd1b311e38bd
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Wed Dec 7 13:09:18 2016 +0000

    Add file loading tests; minor fixes

commit 487d6a7a78877d03e875e8e342f5cde557ddf575
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 7 12:13:04 2016 +0100

    Fix tests; minor formatting (#1)

commit 0e3e0440d05f784d6e92ddac37726776cec05162
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Wed Dec 7 10:10:35 2016 +0000

    Add time-dependant constraint tests; minor fixes

commit 4dede2bc5a25248a61422e58f9f3f659d08b3d03
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Tue Nov 29 11:29:03 2016 +0000

    Update e_con constraint to ignore conversion techs

commit 3cfb9307c888fc4cef0dcc917d442804b7cf834b
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Thu Oct 6 11:57:51 2016 +0100

    re-organise core.py timeseries initialisation

commit 118d80ce4b643b11f22a30b64b5c7b527a2aad07
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Tue Sep 27 12:39:21 2016 +0100

    Minor fixes

commit 01bd883c80e863b7b74e27765cbca1871df69915
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Tue Sep 27 11:27:24 2016 +0100

    Load revenue from file

commit d63e74518c55f1dbb33da771beca9234bbdd2e19
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Tue Sep 27 10:32:49 2016 +0100

    Get time-dependant costs from Pyomo Param

commit 54d3a45aeca4b2370bb5d433393d3e0a9cf1c343
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Fri Jul 8 17:55:24 2016 +0100

    More robust logging of timeseries parameters

commit b170b02ebc697a5301a4fb58618bd07bdcdc3613
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Fri Jul 8 14:14:10 2016 +0100

    Add time variable costs to constraints

commit 7abb6166873aac4dfef33dd5275689b8f5652270
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Fri Jul 8 13:55:22 2016 +0100

    Allow loading time variable costs from file

commit 4a04442f5c71c98caa1d363fcf35eeacbac66286
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Fri Sep 23 20:26:55 2016 +0100

    Get time-dependant constraints from Pyomo Param

commit 791ec82c355d4699ca156e25de1a3e46f9ba21a3
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Thu Jan 5 18:19:02 2017 +0000

    Allow greater range of timeseries parameters

commit 7f65a00b57c0fee2c8fafa7741ff0b81a54f3cd9
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Thu Jul 7 15:47:51 2016 +0100

    Update get_any_option time dependency

commit 28cac0719d73ffcdbbe05477bd7d8d415ad0a3f4
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Sun Jul 10 09:35:39 2016 +0100

    Differentiate between cost & rev classes

commit 87859d3f94ec9009a9a89e7f0c11049e1d5f9241
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Fri Jul 8 14:09:16 2016 +0100

    Add demand set to sets

commit 9eab278a324dc30fcea19a7591a681a7fbce67ea
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Thu Jul 7 16:25:49 2016 +0100

    New revenue constraints

commit a1b0e7c8cf991f683c8647454ca834599ac86424
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jan 16 15:01:39 2017 +0100

    Update version

commit 90ffb1a87081a566c8bb4925d0766e84cdb61d44
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jan 12 11:40:27 2017 +0100

    Release v0.4.1

commit 8d83057920b7b2405ba955bd9987f71326802450
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 10 18:01:37 2017 +0100

    Add Anaconda badge to README; update changelog

commit 30d2c37b894de8611de57cf5d88f2a34a152b485
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Tue Jan 10 16:50:59 2017 +0000

    Add profiling to make.bat

commit 9ed3f8b62cb4c4decf532e6e85b855f806226152
Author: brynpickering <brynmorpickering@gmail.com>
Date:   Mon Jan 9 13:49:00 2017 +0000

    Cleaner definitions of timeseries Sets & Params

commit 0986ad69d7ad1d2aa7ca1608c4e1ff8854dfeddc
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jan 9 12:30:58 2017 +0100

    Minor fixes
    
    * It’s 2017
    * Ignore pyomo imports for linting
    * Allow setting random seed
    * Fix to get_closest_days_from_clusters()
    * Make normalization optional in apply_clustering()
    * Rename normalize() to normalized_copy()

commit 589b062400bac24a831f5fcaee9797cb51a4f82e
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Fri Jan 6 18:23:00 2017 +0000

    Update constraints & sets

commit 7673b0708b47421befa6c9579acd5e6ee43ab328
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 4 14:20:25 2017 +0100

    Re-insert conda-forge installation instructions

commit e39291832d443e5f1753e4bac0537925639dd499
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 3 17:26:56 2017 +0100

    Better printing of version in CLI

commit 7281b29095d40aa6f1da89f2102a0b1824e0d0c5
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Tue Dec 20 17:28:00 2016 +0000

    Allow locations without techs.
    This applies to locations acting purely as transmission nodes. Requires the location to be defined in 'locations.yaml' with an empty 'techs' list

commit 47b64babfca10fac5e6c8c7ff0bc48059f294bac
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Tue Dec 20 13:04:00 2016 +0000

    Update constraint file name

commit a7ae37cfedacbb04bf98210fc3cd59a0b288358c
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Mon Dec 19 16:00:00 2016 +0000

    Reduce cost constraint generation time

commit 91096a11a40b1ce6428314608c8444ca8efd10fb
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Thu Dec 15 18:13:00 2016 +0000

    Rename ec/es_prod/con -> c_prod/con & c_eff -> p_eff

commit 8fc674e5e43db6ffc67774b5191c5573b786dfb4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 9 15:02:04 2016 +0100

    Add profiling to CLI

commit 1c5f00c3f383acbf1b9768968f8fdc9d7e701175
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 9 09:13:06 2016 +0100

    Release v0.4.0

commit b82a137dddc84ec124b2b60300e7f473ebc86c4b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 9 08:50:39 2016 +0100

    Bump Pyomo to 5.0.1; add whitespace linting

commit f500b9616021fd94713a102f4f0ef42e9de2265a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 9 08:43:12 2016 +0100

    Revert installation docs to pre-conda-forge state

commit a4e49c3b7d37f908bafc84543510eec0b4cf5d9f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 6 15:47:20 2016 +0100

    Add make.bat

commit 7c8bc80ab43923d7faa33d62e7723732330c1a9a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 25 08:24:35 2016 +0100

    Configure readthedocs to use conda

commit b9050189a17b52de73bf91ab4af6c62b676ef52f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 24 16:43:52 2016 +0100

    Documentation updates and minor cleanup

commit 1a2ae509f9fd19d032401af47745db86a005d411
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 23 14:37:43 2016 +0100

    Fix missing , in 3a7bbb4

commit 3a7bbb42ad3cb2478adb57c3794078dd709e1c1e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 23 14:28:35 2016 +0100

    Cleanup
    
    * Consistency between setup.py and requirements.yml
    * Minor fixes in core and analysis

commit c9909e5a81b69f303ba41d2c9ce1c4c897cd2acb
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 23 14:21:23 2016 +0100

    Documentation updates

commit f671a2d26ef9625d78160067b05b07091a30e971
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Nov 21 08:43:05 2016 +0100

    Add solver_io option to tests

commit 0b85a2c22f9491a45093b231410ba220ddd9eca2
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Nov 21 07:39:42 2016 +0000

    Get distance from metadata (#23)
    
    * Add get_distance function
    * Add utils.vincenty to prevent geopy package dependency

commit 347ed9e8c12837b95a5224fa987dc3ccbedd68d3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 26 14:24:13 2016 +0200

    Update README and requirements

commit b482c9a807e502d1805c8c9cad9e933f335a7f62
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 21 16:53:34 2016 +0200

    Travis: remove Python version

commit 0b8e0ffa74af55ff972012f3d72805857352a6ac
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 21 16:02:19 2016 +0200

    Travis: fix build matrix

commit 58d147c34ce438be347eec462f89fb6971528e44
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 21 13:53:34 2016 +0200

    Install pylint in .travis.yml

commit afda6a17b0a45a49eb2c1b539350594a30d877c6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 21 13:48:44 2016 +0200

    Add pylint to travis config; clean up code

commit 14ea5b0f772df3a2dc1b0edabc0ee51267614af8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 12 10:27:19 2016 +0200

    Show solver output on problems; better YAML errors

commit bec2b74564d48a176026e77bd3c5800d719171af
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 12 10:24:55 2016 +0200

    README: appveyor, citation info; update changelog

commit 5f15df03d6275cd9f182d852c01272bc98ff6001
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 12 10:14:51 2016 +0200

    Fix to test_io to allow deleting temp dir

commit 3c05eea707c74ecfd0dd06c7077b40a063e7f7ce
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 12 09:55:50 2016 +0200

    Close NetCDF files after writing

commit 6f4a62b73b008c0b03673deb6f2d02e31b323b2e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 12 09:34:56 2016 +0200

    Set delete=False in temporary files for tests

commit 566db2a6661206078b1a294a6391341190295ade
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 12 08:59:33 2016 +0200

    Fix YAML strings in tests; fix travis config

commit 907d238339608475a039944a9fb1c9e52ba6e7c0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 12 08:40:21 2016 +0200

    Remove python-coveralls from requirements

commit f6d1824cf0b27f3b677ca258285e738cc6191e73
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 12 08:35:13 2016 +0200

    Fix escapes in appveyor.yml

commit 677416fb1d7e5f626879ef85a1d43e418d42e74c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 12 08:10:04 2016 +0200

    Add appveyor

commit 0f7375dcaba1854e9c48a36680c5a69934e2b915
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 12 08:09:52 2016 +0200

    Don’t warn about invalid xarray divisions

commit 5539df58700a96b0fb93593712503846268d21c3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Oct 11 10:43:33 2016 +0200

    Clean up logging and core/CLI status msg format

commit d6ee574ff9c7a1fb1145a397640bd3efef40106c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Oct 11 09:34:16 2016 +0200

    Fix issue with max_demand due to move to xarray

commit 399549250ff7f84c98b3361d2604153c2d5e6d42
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 5 08:33:12 2016 +0200

    Add extreme week selection to time_masks

commit bef180838b6ac671e72b2c052f214f52beb5f5ea
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 5 08:31:23 2016 +0200

    Funcs to process parallel runs to single NetCDF file

commit 57cb41c175316fa33440b09931b88552c446a926
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 5 08:29:24 2016 +0200

    Add release checklist to docs

commit 136f715267a5d9105818c18048a2674e5c53f93f
Merge: cf3c149 ec518bd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Oct 4 10:56:52 2016 +0200

    Merge pull request #17 from brynpickering/patch-plot_transmission
    
    Patch plot transmission

commit cf3c1498f0f044f1b4d4f9851c050cb8ed0ac835
Author: Stefan Pfenninger <stefan@uwis-cx-dock-11-276.ethz.ch>
Date:   Tue Oct 4 10:33:52 2016 +0200

    Update requirements

commit 07d645141c16cde6495d8e6590f489f968929b60
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Oct 3 17:38:07 2016 +0200

    Use Qt5Agg backend to work around conda issue
    
    https://github.com/ContinuumIO/anaconda-issues/issues/1068

commit 03e448bc2f772d2c130194936c286b3f95fab624
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Oct 3 17:25:11 2016 +0200

    Set DISPLAY on travis

commit 0cdafd8f834834e006bf84e39b7e3348e622606f
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Wed Sep 21 10:37:44 2016 +0100

    Patch merge error in runtime logging
    
    Few old lines weren't removed on merge during Pull #15 for some reason, just updating that error.

commit f10786cf9f0bb1cdc338602463b5af659317371b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Oct 3 16:47:11 2016 +0200

    Attempt to fix failing test (Qt4 ImportError)

commit ec518bde6de0e8e08fb2e91f6c8f601f201cfc5e
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Thu Sep 22 17:54:15 2016 +0100

    Remove very small positive values from being plotted.
    They will be erroneous residuals

commit 1b9992145afd9290e2f108a6c5dc57bfefee37d0
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Thu Sep 22 12:24:52 2016 +0100

    Adding node labelling in plot_transmission command

commit 04635d2a680bc2476a72413bc28d5fc14d848142
Author: Bryn Pickering <brynmorpickering@gmail.com>
Date:   Thu Sep 22 12:24:21 2016 +0100

    patching error moving from pandas to DataArrays

commit ddc89fecd3da1f0c33d2b036b0867b566f318b15
Merge: cedc236 7bf06f3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Sep 15 16:27:56 2016 +0200

    Merge remote-tracking branch 'origin/master'

commit 7bf06f3d0b3d3c2c17a4a173a70aa26292769d85
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Mon Sep 12 07:50:59 2016 +0100

    Improve runtime logging (#15)
    
    * Added more runtime logging points
    * update runtime logging to remove metadata errors

commit cedc236f5e148b962ba7930d584adfead72567f3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jul 29 09:21:12 2016 +0200

    Raise error if e_eff is file and e_eff_ref missing

commit f0c39b9d63fb04b16eab5fdc629476d96d985221
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jul 28 15:26:07 2016 +0200

    Minor fixes

commit 90237d8459715594a4e72b2611c29eb194f80bfa
Merge: 4a6767c 554d09e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jun 15 15:28:01 2016 +0200

    Merge pull request #13 from brynpickering/patch-2
    
    Update `plot_carrier_production`

commit 554d09e50f3a43be603ae1a963165449edc1a6d7
Author: Bryn Pickering <bp325@cam.ac.uk>
Date:   Fri Jun 10 17:49:55 2016 +0100

    Update `plot_carrier_production`
    
    requires specification of `demand` when running the `plot_timeseries` function from `plot_carrier_production`, otherwise it will fail on any carrier that isn't `power` (as it defaults to searching for `demand_power`).

commit 4a6767c6379045fed2e1a4daf4523d023c472021
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 20 17:25:39 2016 +0200

    Bottleneck/numexpr for speedups; Pyomo 4.3

commit dd2e66f101dcff19aa5e605669bf279a01ad5759
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 20 17:24:02 2016 +0200

    time: closest days for timeseries >1 year

commit fec2f62104a37f703341e4732e0a4f094f0d6750
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 20 09:01:17 2016 +0200

    Bump version

commit df1897be6bc463400da576f5bc361f4166336025
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 20 08:48:58 2016 +0200

    README updates

commit 86551d493abe27c28589cfee9d167a3411fc8cd1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 20 08:37:31 2016 +0200

    Improved time_masks.extreme, fixes to weighting

commit 6e44a3a275248ea532d9e3d935c741f1886d436e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 19 10:49:28 2016 +0200

    Time clustering: work on per-tech basis

commit 482052f104797a40ed34d81f04a48bdcb8b0ac7c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 19 08:14:58 2016 +0200

    coverage to make test

commit 9e9be4d9867c939fa9f9e32b79deb308175674ce
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 19 08:14:15 2016 +0200

    Improved attrs in solution; CLI fixes and tests

commit a3b9ead2616f86354b7b24d34e427633babbb41f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 18 20:11:39 2016 +0200

    Revised requirements

commit 25d91f72409ed573f384685564ce2d5483031469
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 18 20:09:13 2016 +0200

    Cleanup and additional fixes

commit f662ebabd289fa92c3080a1d675d3b7da2616f07
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 18 16:43:09 2016 +0200

    map_clusters_to_data: allow taking closest in data

commit f699bf1217eed2830605f64f4ac1c8759196c5cd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 18 16:41:15 2016 +0200

    Dropping timesteps: distribute weight to remaining

commit 59a3d8bedaa1ad7aa07353e3ca3b31911d55e173
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 18 13:46:34 2016 +0200

    Fix to time_funcs._combine_datasets

commit 376dc4ac635bf527b9a790f91b69a29c4f553eac
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 18 09:13:23 2016 +0200

    Minor fixes to timesteps and remove availability

commit 47a49c4a7b20aee0c902a25e0ddad2a16e05fdff
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue May 17 08:35:16 2016 +0200

    Refactoring of data internals

commit b4c3830ad48ad86dfe8ea1dfdc474ad24aa70b1e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 12 16:09:22 2016 +0200

    Simplify requirements and travis via conda-forge

commit a7470527f173be35eecec1cd44884cd5f69d9671
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 12 16:08:41 2016 +0200

    Minor fixes; apply weights to totals in solution

commit 115d41e9718acc290e6d57a0f2f43a7d68976357
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon May 9 10:40:17 2016 +0200

    Add weights; minor refactor of variable costs

commit 2e557c2b82a6789d8f79708ff9708d165b7f9d73
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 7 17:16:39 2016 +0200

    Additional fixes to new time series implementation
    
    * tests passing, except opmode

commit 20695f495765be2d2af738d951ffc7cec51be241
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 7 14:19:17 2016 +0200

    Remove time_tools

commit b0176c4d78a216dde4a37eb3abf4d9f8379357dd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 7 14:14:53 2016 +0200

    resample: correctly combine with masked data

commit bea0de35ea8c6473e7ea6ecbab1c0a6eccaf9314
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 7 13:22:10 2016 +0200

    Add resample and drop time_funcs; further cleanup

commit 2cec019403588987f8a4b0a380256892c847be1c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 7 11:54:11 2016 +0200

    Time series simplification: Fixes and cleanup

commit af1657bc8e3ffa8d91563f7b684dc9c085163ec9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 7 10:51:09 2016 +0200

    Steps towards simpler time series functionality

commit 28ef939e92a8d239a7d4aec6d2936a04515d2f16
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 6 15:07:25 2016 +0200

    Add logging to CLI

commit ae6e0b329324f56fb59df6849a272122ef0a8488
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 6 15:07:03 2016 +0200

    Function to apply clustering, add weights do old-style data

commit 70e86d356fdca8b4da7788c3a418782d823eea55
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 4 11:40:10 2016 +0200

    Clustering additions
    
    * add hierarchical clustering
    * clean up k-means
    * clean up functionality to apply clusters to data

commit 681fc77ca5e1bbaabbd3a23120abbfc96c7538e9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 4 11:39:16 2016 +0200

    Numpy 1.11 is minimum requirement

commit 62bf3d308d4e64a5ae0ffb4742bc47db4d62d0f4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 14 15:01:00 2016 +0200

    Add time functions and k-means implementation

commit b8135879a97d0d8ad2710b73cd7371bf57ed4af3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 13 10:37:23 2016 +0200

    group column in groups to string rather than bool

commit 9c741423e16188b74344815159177d4eab1c2542
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 13 10:17:45 2016 +0200

    Update installation docs for new requirements

commit d7e0fff6371d4bfdd4c6f30ab8ca12e8f50f947b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 13 10:07:39 2016 +0200

    Clean up code and docs for hdf->netcdf change

commit 1c10850368790c77f0dfacee16a32b73b1dfa0d4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 6 10:03:02 2016 +0200

    Set matplotlib backend for testing

commit 83ba1ebf8283d9aa8f7f48280fbddb67c8ecdbca
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 6 09:45:19 2016 +0200

    travis: use pip rather than setup.py

commit 9f5a4b3928dbec30d3055260d4eb16b37cb21780
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 6 08:51:47 2016 +0200

    Remove tables from setup.py requirements

commit f40eeaf9565c4db771a000a8096ceb885533b344
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 6 08:43:52 2016 +0200

    travis: Fix conda env create command

commit 4365140bd27df9908f070e5e2432299997473d33
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 6 08:35:14 2016 +0200

    Use conda for travis

commit d3820a5835ccc63bedca4d6b1021512459bedd1b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 5 11:31:22 2016 +0200

    Replace pandas Panels with xarray DataArrays
    
    * Solutions are now xarray Datasets
    * Replace storing raw HDF5 with NetCDF4
    * Simplify CSV storage
    * Improved tests

commit 4b92e11addc12126e861dd92f78f5c5248ceaae2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 30 09:41:32 2016 +0200

    Pandas 0.18 compatibility

commit 1394e098c3be6d5cf74785dcb9a39382c5803e17
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 10 16:22:33 2016 +0100

    Release v0.3.7

commit 7020fc572e4597700f8d7e2520044ee41f8fba79
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 10 15:42:08 2016 +0100

    Cap pyomo and pandas versions in setup.py

commit 4aa5a67c7e2038cebde5a2e327b024e0ef4c4260
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 10 15:41:36 2016 +0100

    Solver options fixed for Pyomo 4.2

commit c0f5676d464f2e39f7521190b9b4206e5104e2d6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 8 17:31:20 2016 +0100

    Extended per-location overrides

commit fb7d1b55da62354847911ca70f0c91e8efd1483d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 8 17:30:09 2016 +0100

    make test

commit 6be165cad8628149eaf9f8afe4c8d3a58acdcede
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 8 14:01:52 2016 +0100

    Error handling improvements
    
    * hardcoded output.path default
    * log warning if attempting warmstart with a solver not supporting it

commit 8f606a6ee52d4992084fa4c5ec6cff824af89e21
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 8 12:02:20 2016 +0100

    Improved installation instructions

commit 10e3fb7bf33685967063ae1b167b0c0062291431
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 8 11:12:41 2016 +0100

    .cache to .gitignore

commit 643d0396149daee82f338869f47a57e8ef1f653b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 8 10:11:20 2016 +0100

    2016

commit 28cf19f261d937b224fafad9880992bd1b2655a6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 8 09:33:55 2016 +0100

    Clean up legend funcs in analysis_utils

commit a4ebacb355f43d973cd0c9c9a0ed73c0e2c0f10a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 8 08:36:27 2016 +0100

    Python 3.5 as default

commit b05928035569c13ae8a1190c0fa9a0674ae29c8c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 2 15:46:30 2015 +0100

    Raise numpy, pandas, matplotlib version requirements

commit 9a04cf33123d00ea58ebca4d2e719ac11a6447e7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Nov 30 15:29:29 2015 +0100

    Updates for Pyomo 4.2 API, pin pyomo in requirements.txt

commit 29e39ccdc6490c20d47bcb45789a7111cd754f0b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 25 16:27:48 2015 +0100

    coverage==3.7 to requirements

commit 815e8d5ecadb1b0464a0517b42d69bdbb92b8e81
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 25 16:27:33 2015 +0100

    Workaround for non-numpy divisions by zero

commit a9b07e8eb0427a1724d900c29bab3148c9256c09
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 24 15:44:13 2015 +0100

    Add builtin model test; proper Zenodo README badge

commit 297bf2cf09754cbdb357a2cfc6fea4de52b87ea4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Sep 23 10:09:34 2015 +0100

    Release v0.3.6
    
    * Fixes to tutorial

commit 3604d97c5ff5e81a37f08db6985a198bcd44f789
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Sep 18 13:32:54 2015 +0100

    Release v0.3.5

commit fa667269f789a3bfe64d2da772823ce99116c578
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Sep 14 15:20:39 2015 +0100

    Fix RTD by making pytables optional in setup.py

commit 400607a9e234812197594566893c2ea898e23cd4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Sep 14 14:46:54 2015 +0100

    Bump pytables, fix pip log clearing for travis

commit 79555d15029ce855a9682608a5194243dfc8fd02
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Sep 14 14:39:00 2015 +0100

    Minor streamlining of time functions in core

commit c6b96a371327802e8e17a579aa85753fbf12758f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Sep 14 14:26:06 2015 +0100

    Various minor fixes
    
    * Bump requirements.txt versions
    * Fix flipped and/or for masks_to_resolution_series
    * Add as_frame param to analysis.map_results
    * Remove pip debug log after travis builds so it doesn’t pollute cache
    * Minor cosmetic fixes

commit c25aa6dddb3b4c05d7cc3042fcc7289533bcf38c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jun 1 13:09:17 2015 +0200

    Minor changes to analysis

commit e0421e6e91cd86773ec4d7a00c5a18df2809f9ee
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 15:03:23 2015 +0200

    Fix 41475c9

commit 41475c9033d9ed1d99c2fe419531eface190ac6c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 14:48:29 2015 +0200

    Allow dropping time steps

commit 7ea821a956800102a853bcaf8d6f41ffd9503314
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 11:28:34 2015 +0200

    travis: no caching of calliope binary

commit 67058b4434a515af843ac42f8e6278ed6e02bba0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 11:10:27 2015 +0200

    Minor changes to parallel section in running doc

commit 9559bf746c70573bc9593da51d1f7398f4fb32a3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 10:49:45 2015 +0200

    README and doc/index harmonized

commit 649a941eb7bfa2079aa22ff438251971e994b457
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 10:42:23 2015 +0200

    Minor fixes; documentation updates

commit ca49a031cff96e020c2fc6726fc907e4a4b27be8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 09:58:48 2015 +0200

    Preparation for multi-level balancing

commit 0285490c1616c65e9d88d25ca2de049c46d91ac1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 09:09:02 2015 +0200

    Doc cleanup

commit 75866919aa751c4d6c6b28900001b44e95949636
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 09:02:39 2015 +0200

    Dynamically load time mask functions

commit 1e37228a24c5ca35edf6f17d08145339e0ad3899
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 09:00:19 2015 +0200

    Add analysis.get_levelized_cost

commit 417f5c3965b561fbf68cf8e20054420f00078da5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 29 08:07:31 2015 +0200

    Analysis additions

commit 0c2e652d091828f3c903e1da4f41688b4e2035f4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 28 15:18:03 2015 +0200

    Use sentinel in AttrDict.get_key

commit d034a23b30d59dfa0ae530b80bfd35b35917add0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 28 15:03:04 2015 +0200

    conf.py cleanup

commit 3641ba019b624ced95f1991257ab493bd43167a4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 28 15:01:39 2015 +0200

    Remove level config (auto-generated), add location option inheritance

commit 7eb0aea0a9f810d30cc330b697f46bb3eb5f9084
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 28 09:17:38 2015 +0200

    Makefile update

commit 090a4b7277f3f9fddfa5d962c1992e3fe9280ae5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 28 09:15:08 2015 +0200

    Revise capacity constraints configuration

commit e10ab52122c2dbc659ead389eb355e264b85d104
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 27 15:45:30 2015 +0300

    Add e_cap_max_total constraint, group optional constraints

commit 08acfbbc1bc91fb764df9780898bd2962908dde7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 13 09:02:04 2015 +0100

    Improve get_summary, fix minor CSV and LCOE bugs

commit d1bdb6ba61524c978c1caae8c48337cf1e68ce7c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 8 15:36:02 2015 +0100

    Force r_area to 0 if e_cap_max==0

commit fd525d1baf29d4b83a982d8f7d96115604cc7212
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 8 15:26:03 2015 +0100

    Fix bug in operational mode when using rb

commit 2a9086298b187b99004ae1624c67eb3fb40602a7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 8 15:25:40 2015 +0100

    Improve generate_constraints

commit 7620c0652b29a95f318b7dc4a17b583ef1f956be
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 1 11:10:56 2015 +0100

    Fix README links

commit 49bbf45fd5e13659075a7ca8f5c1a840beaf8a72
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 1 11:08:21 2015 +0100

    README improvements; citation info; roadmap

commit 2ebf398cb84d9550621822e3660f566b41b0228b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Apr 27 10:16:32 2015 +0100

    Release v0.3.4

commit 51c0860ae94e37637cc517689a3a2483238df8ff
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 24 08:43:53 2015 +0100

    Add openmod-initiative link to docs

commit aa15da9843d7105207ca20220aef55ac9ffd944c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 24 08:43:08 2015 +0100

    Fix bug in operational mode cost calculation

commit 8abfe9167e543e2c6c263a473ea2c2180306a745
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 3 10:17:59 2015 +0100

    Release v0.3.3

commit 25c8f032136ba13e1be628664f1cc519ac1fee83
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 1 17:20:46 2015 +0100

    Don’t resample if already in desired resolution

commit 0d0a13c2f0226dec1e6b3e276fd29aff13293b4e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 31 15:27:07 2015 +0100

    AttrDict improvements, --debug parallel runs
    
    * convert dicts in lists to AttrDicts too
    * improve handling of nested keys
    * add `flat` parameter to as_dict()

commit b5ab98962bc8de54aada4748da2c0872854fb310
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 30 13:22:46 2015 +0100

    Docs: minor improvements to formulation

commit 38d746baf13730bbdd5b53f91087b0c511c17b72
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 30 12:02:24 2015 +0100

    Additional data consistency checking

commit f21b7c113b5482a0038b376d8ea3cd4e3e92fa72
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 30 10:25:06 2015 +0100

    Analysis: remove hardcoded units

commit 99d79d9351e88b19b3965c658823258132ece504
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 30 09:27:53 2015 +0100

    Travis: use container-based infrastructure

commit 9c122e2bb60f60bb3a18a10ff06a6b108a72d877
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 30 09:23:19 2015 +0100

    Example model unit fix

commit c21a0e03f663669cb7ef6d6caaca9d15021427c9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 27 10:24:45 2015 +0000

    Fixes to time masking

commit 5f46ce3aeca7222eb8db7f1a98202b16cdfd230a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 20 16:24:06 2015 +0000

    Refactored time masking

commit 278b906f6039cef598bff03fdad0a9bb12dba343
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 20 11:41:31 2015 +0000

    Moved mask to series and back funcs to time_tools

commit 8c9cd171fb3aeaa8b6b125ffff6a50612cfc0fc0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 20 11:40:18 2015 +0000

    Minor documentation updates

commit 7552bce4ef8091edc60f0b43d12f2ac722804546
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 20 09:24:24 2015 +0000

    Flip location level ordering (0 is now above 1)

commit e3fccc7837cd1c68e66f4e18b53edf2cbb75a180
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 20 08:44:59 2015 +0000

    CLI: Show preprocessing time

commit d333bf77933d660890e0bbcf9a56f5cbfe8de610
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 16 15:45:35 2015 +0000

    Save processed time series data in solution

commit fa504e0009a067511e82dce647b793f1a13d90d4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 16 15:41:17 2015 +0000

    Change how storage is handled in c_s_balance_pc

commit 0cd606acc8f1bee9281f35ad7b8a1bef4e595171
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 16 14:58:14 2015 +0000

    Small fixes
    
    * Allow building model config without reading data for debugging
    * cost_types option for utils.cost_getter
    * More resilient read.read_hdf
    * Moved map plotting to analysis_utils
    * Global default for costs_per_distance e_cap

commit 4d7f74a4057ba1b45bb03d4bb65dfa1a23ece9a3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 16 14:54:01 2015 +0000

    CLI nicifications

commit 0876e0cd79ee0b406de3cd0cc4c5ea0c8306fdf8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 3 10:38:06 2015 +0100

    Add model/run names; more verbose `calliope run`

commit d04e2ca648f71f567d9fc94471611526290fffea
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 3 10:14:16 2015 +0100

    Check for location using undefined tech

commit dbf3cef5af44b3dba7c9b92acc1a52b3f11b9c16
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Feb 26 17:01:31 2015 +0100

    Check for invalid string options for r and e_eff

commit ca5a51df264f42fe8e7883a8ba462dc79962e5f5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Feb 26 17:01:02 2015 +0100

    Add pdb option to cli tools

commit 54b87b2758b52bf3560e9259fc9dfa46b85c8bf0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Feb 26 16:57:10 2015 +0100

    Windows install notes

commit d07e1b65b28ee93335b4e6231763be3791f16968
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Feb 24 10:48:01 2015 +0100

    Fix 5d499a5

commit 5d499a5bcb717a1a107d9360aa8a6c806cda87ed
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Feb 24 09:45:09 2015 +0100

    Rename and better document debug settings

commit 5f03e2e843c8a15354937c9acb278dd20a68a5ee
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Feb 16 15:07:29 2015 +0000

    Fix documentation build problem, other minor fixes

commit 0abc9e243102212bbc9b9fa1c89df92208ccb6ca
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Feb 13 15:42:09 2015 +0000

    Zenodo badge fix

commit 3202b812541d423060b3bbc7bc9a93259689454b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Feb 13 14:42:35 2015 +0000

    Travis fix for Cython compile issue

commit 49a161eab17e493485849765c88f70a7249c1c84
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Feb 13 14:42:16 2015 +0000

    Clean up analysis/output modules

commit cc618a5afe691d674c26500164fc9abd4f0f59a1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Feb 13 11:59:32 2015 +0000

    README shields update

commit 8fdb3e488023db79fdf9ae8f237cc937d6dbb25e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Feb 13 11:16:56 2015 +0000

    Release v0.3.2

commit 851fb781c269446467b31c63fef3f2e37cb2bef2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Feb 13 10:33:35 2015 +0000

    Solution col names match model components

commit 61b9d23817eb79e2401aabf498ee9e1bdc44fab7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Feb 13 10:04:35 2015 +0000

    Streamlined time internals

commit 5bd35eaffde872a4fab5ede8253d90d9bd07c14f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Feb 12 14:50:13 2015 +0000

    Minor core fixes

commit 80e421478859e21b1bdf76a02b2cf387e90772be
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Feb 12 14:38:37 2015 +0000

    Allow saving to multiple formats simultaneously

commit b6ef1195d0b665051e15b47269d590034bdeb01b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Feb 11 14:26:07 2015 +0000

    Unify overriding, allow _REPLACE_ key

commit 726a547d38f2998ec24eaa43fdebfc3e75325ffe
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Feb 11 10:44:23 2015 +0000

    Simplified time_tools, fixes a subset_t bug

commit 369145f42c7b78ff2a8da508220d8cd5ae21fbfc
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Feb 5 09:38:51 2015 +0000

    Specify Python 3 only in setup.py

commit db497e68b7c3ffb984a0a3caa5390e81d689be2c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Feb 5 09:38:23 2015 +0000

    Change to ModelWarning formatting

commit 76d2e7a4552f4d31b1bfc812fe6913305a3de9a2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jan 29 14:54:35 2015 +0000

    Index checking for loaded CSV files

commit ac1240631df6df5ff147e62ff0d9e81f8be93423
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 28 09:17:12 2015 +0000

    Don’t save locations table in reults

commit 635eae475dc30e7872f2f3e0a47f18f2fb022ad2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jan 23 09:21:49 2015 +0000

    Cleanup CLI

commit 43794b67bbdaf007c837c131d5970ea754c6ef80
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jan 23 09:15:57 2015 +0000

    Add model_override setting

commit 8fdc863cc0fe3fd88f68796ccd0ee1a16d7e05a6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jan 22 16:39:40 2015 +0000

    Fix f7b69aa

commit f7b69aa97a717f464e0faad0f34ecf817ac8989f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 21 14:22:31 2015 +0000

    Parallel options: pre_run and post_run

commit 09b8fe8b310be4d32eb0bb9116fda842d9b7c79e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 21 14:05:16 2015 +0000

    Added save_constraints output option, fixed YAML save bug

commit f8c602852b853fa740a6a8453651f157f406fcc7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 21 09:54:27 2015 +0000

    Better error handling in cli

commit 2d894724e880b443f7820115f482d59e0eb9496e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 13 14:14:11 2015 +0000

    Revert travis changes

commit 60f8d4159f6d13ab7de478f3ba6bb1148eb3232f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 13 09:28:03 2015 +0000

    Fix 79e4aab

commit 79e4aab741f6bda56b502ed701db380fba7e5847
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 13 09:02:03 2015 +0000

    Install some deps via apt-get for travis speedup

commit 8c5f31317628a4c2d87dfe85ca9242897738be5a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 13 09:00:47 2015 +0000

    Fixed bug when using mask from file and subset_t

commit 02201abca1c7c79639c57ea2f5afb58da00fe763
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 6 13:06:27 2015 +0000

    Fix ae1d216

commit ae1d21611515086fc6be1cf74c030d6a3963131a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 6 13:03:06 2015 +0000

    Release v0.3.1

commit 5f018bcbc29f2b95324b6669cfd97ef7c27e1a03
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 6 12:40:45 2015 +0000

    Pyomo 4.0rc in requirements and setup.py

commit 554f2f596f1bb88ab36475ed7aaa6b79e90309c4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 6 10:22:22 2015 +0000

    Flat style!

commit 081ec02e8a3ef1a4e1ddc613127a298248d793ab
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 6 10:17:03 2015 +0000

    Logging improvements

commit a463cdb454e83044967c5c1c4274d0579ed6fd6d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 6 10:14:47 2015 +0000

    Install docs fix

commit d25db8f5b42578685c45de4f9ae8897d1f752943
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 6 10:14:20 2015 +0000

    Enforce inheritance from built-in techs

commit 3cdc903e3399f15d66bae5433862d65f738225ab
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 17 09:27:02 2014 +0000

    README update(1)

commit ad9aa66f6afb5d815bde3c3753621df2ed91a029
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 17 09:25:30 2014 +0000

    README update

commit c91c40ce3ca34d2333a909c129a18f1711c6be33
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 17 09:23:26 2014 +0000

    Fixes to time_functions

commit 36568de7fd085cab6ef0432eca517b19a953ef70
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 12 13:05:27 2014 +0000

    Release 0.3.0

commit e63ab0fdcee2769d5b42f81201680a8ed6a6722c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 12 12:52:51 2014 +0000

    Added example notebook

commit 02beb321a09ae21a40c855cb6f40f49918d38bb2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 12 12:46:49 2014 +0000

    doc/requirements fix; other small fixes

commit f98cc7cf4381180ebc4eb908044dd874e2555039
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 12 12:16:17 2014 +0000

    Docs: formulation fixes

commit 2f67f064a4783547e9453639b32a1c17081224d8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 12 11:48:38 2014 +0000

    Ran nx_pylab through 2to3 and removed unused funcs

commit f17994426f5536fd0027acf851aad03a51a2735f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 12 11:01:45 2014 +0000

    Documentation updates; various minor fixes

commit bad837ffa40e67cafc97325e6c108a157c87e396
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 11 10:22:50 2014 +0000

    More minor fixes

commit 20f0e422a6a7ccc2aed972ec4d1078c1973aba26
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 11 10:19:01 2014 +0000

    Better display formatting for warnings

commit de086c0949db5e5fba1a07a96ee5f2d518acbd4b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 10 10:39:42 2014 +0000

    Minor fixes
    
    * Use warnings
    * If no stack given for stack_plot, use columns
    * Don’t double-count c_eff for conversion/transmission

commit 96531a7c0a13cb2a7ef0c9588e01e72999264ff8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 9 08:52:26 2014 +0000

    Moved repository

commit 51448a0d10f9114db9777a5ff696f0b10af4efd5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 8 17:13:48 2014 +0000

    www url

commit 8a58fb7ffd1a9fc35017adf049e4ac77bd155542
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 8 09:10:38 2014 +0000

    .gitignore fix

commit 5fbcf8334f8179188dc4e36a1ffbcb3353bad8af
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 8 08:50:48 2014 +0000

    Fixed trunk checkouts in travis

commit 37003085df2a7e737448df3e29d85c31a4b3985d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 8 08:48:59 2014 +0000

    Small fixes to docs

commit 4b84b2b04674487deb1d37ae51d704c6accf5c31
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 5 15:20:25 2014 +0000

    Remove u string specifiers

commit 56559ba46245ad832b8e281c54de6419a7560264
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 5 15:13:56 2014 +0000

    Docs: more updates, including intro

commit 9ef64c352f70221f3caec69842165371034781a3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 5 12:55:59 2014 +0000

    Docs: parents and groups, other fixes

commit 507c1ed70218062c662b3b83886462e79d362360
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 5 10:04:54 2014 +0000

    Docs: some analysis basics

commit 9871d87adcb0febbcb9d1929d33fe1b6b02107f6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 5 09:45:56 2014 +0000

    Further doc fixes, including brief dev guide

commit 734b5468b05105d629243a5d1fc320123d940ac6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 4 17:54:53 2014 +0000

    It’s 2014!

commit 20a46ebb5d6288e7d6eaabcfd66694c935454db5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 4 17:35:44 2014 +0000

    Docs: running models, API doc fixes

commit b231827cc92aafa30ec925aea863c64b6f7245f7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 4 16:55:41 2014 +0000

    Docs: run configuration, YAML format, other fixes

commit d570c12d048a09118826a77d849de8995a047c17
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 4 15:48:19 2014 +0000

    Travis: track Pyomo trunk temporarily

commit 6ead5090430ef15bd0846cc4e68c27335aa9c3f1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 4 14:51:12 2014 +0000

    Docs: configuration section, other changes

commit 0e2374851cc95f140bc5ea9c995330e26ddde30e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 4 14:49:45 2014 +0000

    Documentation theme changes

commit da3eccb4576a5037fb2d8a8ba6facb4c2e907bfc
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 3 16:37:01 2014 +0000

    Updates to configuration reference

commit fc8d56e97537adf217c625f76c629cb8e878f98c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 3 15:13:08 2014 +0000

    Drafts: partially auto-generated config reference; example model doc

commit 89a134bba43a9af194ccce25b445b5562adbbbd2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 3 10:14:05 2014 +0000

    Doc design updates; revised formulation section

commit e6ab4ef000720719fb9735ae5b09308f0381102d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 1 17:01:13 2014 +0000

    Updated components doc, more structural changes

commit d708c040dc7eefb2f384be9d48a3cb35b4f4805a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 1 16:44:29 2014 +0000

    Better documented example model configuration

commit 2c7309da7a1acf53444a20c33b00ca908359e72f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 1 16:43:59 2014 +0000

    Documentation structural updates

commit 1919467dd95fbdb448b5ab4bc7916eb3ba7cae9b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 27 17:36:17 2014 +0000

    Be silent about matplotlib import error

commit 718ee39ba49c46de969f721156e4969ca16fd4fb
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 27 17:24:16 2014 +0000

    More fixes to path management

commit 6d33dd83d0aaf8adae68c3c10877ce084d0b1a77
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 27 16:20:29 2014 +0000

    Analysis imported in core only when needed

commit 337dee75593a4e97b38c4e8e4b62d465270de133
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 27 15:37:33 2014 +0000

    Fixes to parallel run generator

commit ba68a00073dee67c8c3104523abeb62d8681da06
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 27 12:48:33 2014 +0000

    Data path now in model settings, cleaned up path processing

commit a72ea2644d5871741ee8a7a8966e529da0820170
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 26 15:18:45 2014 +0000

    Use functools.lru_cache for @memoize

commit 45775b7b2cddb14151c920b975c8bcc009f659f5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 26 14:49:00 2014 +0000

    Documentation structural changes

commit c2efd702fd6fdccc69af9b119d300b5b53fa571b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 26 09:53:05 2014 +0000

    Small fixes

commit 5b15002c9f231a68299e79d4a7ab51db01b7d31f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 26 09:52:50 2014 +0000

    Notebooks to .gitignore

commit a75992f8a5229337e269ed690e24fe0f85d36a1f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 25 11:51:13 2014 +0000

    Python 3 and Pyomo 4 (no backwards compatibility!)

commit bb5b753b5f18fdf9c7036715f1143fab5f7d0c3f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 20 13:45:35 2014 +0000

    Store entire solution in a single HDF5 file

commit 31dd88a2a161a9ca65906f7b95a5ce4db3afbcff
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 20 12:43:35 2014 +0000

    Improved planning-to-operational functionality; AttrDict fixes

commit b46d476ed22846397070bb85c7e8841317dc88a7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 20 11:01:11 2014 +0000

    Added DummyModel to analysis

commit 2cbe33f745e67a5039ff3bb9f9d5cc7b43e77807
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 7 14:32:01 2014 +0000

    Minor improvements to stack_plot and legends

commit a39ac5aa0cc38573ce65dbd16f8c4a07f64223cd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 6 17:11:15 2014 +0000

    Improved option_getter

commit fde4896a896d996069fc98187a90c27a5fa2a3f7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 4 12:03:03 2014 +0000

    Updates to base constraints and other fixes
    
    * Better parasitics
    * Better secondary resource
    * Other cleanup in base constraints
    * Exclude tests from code coverage

commit b9d1a22c019bcd8b5c9b057acfdb8d75027e9827
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Oct 28 07:57:37 2014 +0000

    Improved parallel run generator
    
    * Always write a unified run.sh file containing all iterations
    * Auto-detect single vs array style based on whether there are
      any per-iteration parallel.resources overrides

commit f4d5f7253af60b55a4ddba6264fbcc9496024f38
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Oct 27 10:14:51 2014 +0000

    Improved c_eff implementation

commit 3603411b7e196d311a6f7e60ceadf8b2261e7b36
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Oct 27 09:16:54 2014 +0000

    Fixed time_tools loc index

commit 4d0d3d889103c34f8ca903bff233dd9aef47e5db
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 24 18:02:23 2014 +0100

    Don’t insert defaults in generated parallel models

commit f3fd6e828ffaef349d9f144747ff0da5e6690d35
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 24 17:46:13 2014 +0100

    Fix instance checks to work with unicode strings

commit d60f9269240ed73554f50ca59090a75f95515821
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 24 17:18:55 2014 +0100

    Default arg fixed in generate cli

commit c53b168247163240129b57742f097f8d426dea14
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 24 16:55:20 2014 +0100

    Added c_eff

commit 13394470e62d9fca5093f192ca94c97f2581f835
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 24 15:29:45 2014 +0100

    Core improvements; log index if constraint error

commit 43ab68cb3a2dbf4cdef463bf60e9c4041c4e42d5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 24 15:28:57 2014 +0100

    Small cleanup

commit 9213ec2d8e778ea940ba733ce1a6240e12485116
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Oct 24 15:26:38 2014 +0100

    Fix speed regression in time_tools

commit 174ac8b26c3a23a262c9cf78e52c652547c5bcab
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Oct 23 08:58:20 2014 +0100

    Removed use of eval

commit 26766cf469469392fdebffa0ec05fb6d90b58637
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 22 17:40:56 2014 +0100

    Fix failing test in 4c19e98

commit 4c19e98d5ea1817fa815e81bf3dd05db7b72ee47
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 22 17:16:23 2014 +0100

    Cleaned up setup.py and added MANIFEST.in

commit e8df9e79029eb347a77f761ef297481033eabb24
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 22 17:15:41 2014 +0100

    Better command-line interface, moved example model

commit 840a389e5e0bd15d467426530f10e4217e0b6357
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 22 12:49:26 2014 +0100

    x_map consistency check and error messages

commit f5ff7d2e8481d9b783d3407210792225c050fd68
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 22 12:19:14 2014 +0100

    Consistency check CSV file index on load

commit d7cab2e3bbe675e2a2551471f9b034e025dc8133
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 22 11:09:44 2014 +0100

    Ensure that default technologies are not redefined

commit 84f528a848dd6666306d724f75c6cc2364114702
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 22 11:06:18 2014 +0100

    Separate loading of objective function

commit cd322938925462ee989cac68b787e660fd10616e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Oct 22 10:06:57 2014 +0100

    Compatibility with pandas 0.15, bump other versions

commit 606ca8c6806ab3e2c971ceaa94c18be24e9c0d19
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Sep 29 16:20:00 2014 +0100

    Moved option getter functions to utils

commit caa87ec6c701ae6908dc2f25650f08b4c23d737e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Sep 29 12:15:12 2014 +0100

    Permit locations without 'within' definition

commit 10a59c404b6bfce4ed2c9ec93e068bd53764277f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Sep 29 12:13:57 2014 +0100

    Simplified time tools and masks

commit c9dbf98551b722da0143763afdcf26356d8a6421
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Sep 29 11:46:59 2014 +0100

    Fix tests for x_map change

commit 9bde45c69a3dbf0bd2108ccd42efa69fec480958
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Sep 25 16:28:36 2014 +0100

    Added how option to resolution_series_min_week

commit af380661e920dcab31efefcc83bbb26dfe0fff69
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Sep 25 14:51:24 2014 +0100

    Better x_map, allowing for multiple mappings to same data

commit f57b02cae752b4248735f6690fe99bc45bcb157c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Sep 25 13:48:07 2014 +0100

    Improved time masking functionality

commit 0e05d28f46e5302febfdc2ae89159cb270eeda7e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Sep 25 13:47:17 2014 +0100

    Minor update to parallel indexing

commit 1d92ead0ea60c358c877bff4781a909cb87f98e9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Sep 24 15:02:11 2014 +0100

    Fixed bug (looking for data file even if tech not defined in a region)

commit aa15caaea6ea1d69b015ab235fa492a74230662d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Aug 6 11:26:34 2014 +0100

    1-indexed parallel settings

commit bef0ceb4380617229392540df53711bb08894710
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Aug 6 08:42:09 2014 +0100

    PyTables installation fix

commit 289a06f5fac42fcf56e080a03a6d9ca913d86fb6
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Aug 6 08:31:36 2014 +0100

    Proper requirements

commit 0a523fd190e51c184e34d0d7aa2e04d06f445622
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Aug 6 08:14:18 2014 +0100

    Fix hdf5 package choice

commit 827deb18cf568f3ecc2074bcf16db33f771fdd0c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Aug 6 08:11:07 2014 +0100

    Fixes to setup, analysis, and qsub parallel env

commit fcf6610a3ba92c4d7e1cb869dbf77024231cb483
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jul 28 18:19:57 2014 +0100

    Various improvements and fixes
    
    * Fix to es_con_max rule for conversion techs
    * Changed scale_to_peak behavior
    * Improved analysis plotting functions
    * Changes to included sample model
    * Other improvements

commit ed33833710a234e7263e16d926b20814feb08e06
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jul 28 08:57:12 2014 +0100

    Fix README

commit d29f2b1ffc988f5c50f46707d55d617fa69dcc4d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jul 28 08:55:19 2014 +0100

    Updated .gitignore

commit 94f719919fc8f2bb97ec615046a781a1573990db
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jul 28 08:54:50 2014 +0100

    Added coveralls

commit 3c238e4be5ab45419e98ce8cc36998e78af28bfd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jul 25 17:20:27 2014 +0100

    Fix d30b209

commit d30b209172454a8eebe536e2b81ef11ce960a444
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jul 25 17:10:40 2014 +0100

    Added numexpr to requirements

commit 44ffe3e975b60cedbee614f0bc7f69fd5bdcd68a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jul 25 16:57:16 2014 +0100

    More travis.

commit 8d0432c537025dee0d3ccb575c5562a5646108b9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jul 25 16:46:59 2014 +0100

    Added .travis.yml, fixed typos

commit 0c47ff890f539617d9295d76b39b9d4c6c5b703e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jul 25 12:33:35 2014 +0100

    AttrDict: changed import behavior
    
    * local statements now take precedence

commit e23ba6135fecd2ca7c846f9780251e0a6d624221
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jul 25 12:31:47 2014 +0100

    README cleanup

commit 85fa78d7505d0a44f9fd0a01f9f3342c5fe96213
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jul 24 19:07:42 2014 +0100

    Documentation download links, cleanup

commit 97cb0403e494b13db9674ce07399603e8b1c4afc
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jul 24 18:28:23 2014 +0100

    Cleanup, minor doc additions

commit 019769e61f8fbbb252d3eb923fdf34ab6a600a93
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jul 24 11:58:17 2014 +0100

    Additional Read The Docs fixes

commit 0177832127cf6ca7c0f0b64ba661b1a09bbf44d4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jul 24 11:26:30 2014 +0100

    Fixes for Read The Docs

commit 87c0681e1011a7aaaef4bacdcced4e7f54e38d4b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jul 24 11:00:52 2014 +0100

    Fixes to __version__ import

commit b01b0e4b7fa3bf5e2c705ba4766f44ffee605800
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jul 24 10:50:44 2014 +0100

    Updated README

commit 43f561f8a1bbc65cc51d28fbfeb3747589fefc75
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jul 24 09:52:37 2014 +0100

    Added demand_power_peak_group to group_fraction, updated changelog

commit 1ad25820dac9b20be013358e17b3c922c87e02d1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jun 4 22:03:53 2014 +0100

    Switched to GLPK for tests

commit 62370776c6d6b5b05e653e7f47abd1dc32c5aca4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jun 4 22:03:18 2014 +0100

    Fix to location ordering in analysis

commit 521e065cdd748295e2296ece99e0ac23101fe22c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jun 3 15:18:20 2014 +0100

    Logging and fixes in parallel_tools and analysis

commit 8e82079d38ef9824756cc74c962380e4e776f3d7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jun 3 15:12:51 2014 +0100

    Pandas 0.14.0 compatibility

commit 45726c3525b2da03eed997541039a75cf6894ff5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 30 09:35:48 2014 +0100

    Compatibility with Coopr 3.5.8669

commit 96c977ef7702e2ccac83fcfb2dab95ba30378202
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 29 18:09:21 2014 +0100

    Added ``debug.echo_solver_log`` option, removed old JSON saving

commit 6d0dcb66de48642be475ec8a77378c02bac272af
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue May 27 07:56:15 2014 +0100

    Further small bugfixes
    
    * Fix infeasibilities caused by group_fraction constraints
      by allowing some techs to be ignored in grouping
    * `unmet_demand_as_supply_tech` to allow unmet demand to participate in
      group_fraction constraints
    * Pass entire model and run config through to solution for later inspection

commit 7b91425fcb732e5f8c50930b4b995c0bae1deb06
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun May 25 12:22:52 2014 +0100

    Plotting additions to analysis.py

commit 78d15ab849a31ab758ea431b31813204508bcf0a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 24 14:44:12 2014 +0100

    Better logging level in time_tools

commit aeff7a2e2284e87a0b6cc935a3f19b6c282f9b70
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 24 13:22:40 2014 +0100

    Replaced .ix calls with explicit alternatives

commit 41de99b4423754a8ab8160b46567ba3fb21f04b5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 24 13:12:40 2014 +0100

    Cleaned up FIXME markers

commit 4adeb323a5144317570b806ded657c1abfadce59
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 24 12:48:49 2014 +0100

    Cleaned up TODO markers

commit 4b632e6e7738271f564905e85210575738bdd1c9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat May 24 11:21:40 2014 +0100

    Fixes in analysis.py, improved get_option

commit 9126511e127020eeaa34555a3ddd29f314c3fa9c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 23 17:32:48 2014 +0100

    Reduced use of ‘total’ row to LCOE and CF solution

commit d74ab13c9b31c2bd7521ecb2da67674c9b7fb72e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 23 16:33:39 2014 +0100

    Tech sanity checking after initialization

commit 19c3f86e4f1d25435dae7b0c8447d9a27311a6bc
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 23 15:49:41 2014 +0100

    Added a tiny bit of logging

commit 26e74299d5486f95c745680cd41f14d022b032cf
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 23 15:13:42 2014 +0100

    Cleaned up exceptions, added custom exception classes

commit 10c46b2e754603a40e6e4d4889a600d7e3d91aaf
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue May 20 15:33:38 2014 +0100

    Catch parallel run definitions without a name (likely shared configs)

commit 45ab8acf6007edecb8d87c17e7cf3b467f768b86
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun May 18 18:32:01 2014 +0100

    Changes to get_delivered_cost

commit c092a670da5e39512f8fcff39c8642902aced607
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 16 16:46:20 2014 +0100

    Bugfixes:
    
    * Config: e_can_be_negative to e_con and added e_prod
    * Fix to get_costs()

commit 10455497b1e92bb8eee6b0de561dd086deb939ae
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 9 11:49:00 2014 +0100

    Added solution_to_constraints to analysis.py

commit 045653f6ad884f4c5c441365c79b76111eb67c27
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 8 14:24:32 2014 +0100

    Corrected cost calculations for operational mode, fixed other bugs

commit 7b45ac6d2359b90e54b05fe68256a2872fd55110
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 8 11:05:39 2014 +0100

    Distance-dependent constraints

commit 8a5ba63bad3b5b6fc58898a7c07a339d873973b5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 8 10:53:54 2014 +0100

    Allow custom colors per technology

commit 88828e3770487f69b3cd107b6ec347c9e79d98f2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed May 7 11:33:41 2014 +0100

    Diversity indices added to analysis.py

commit e02e4d58d9c0a726c3bf8cbf81b9f28b50dcb132
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue May 6 19:34:32 2014 +0100

    resolution_series_min_week added to time_masks

commit 5f300e5ed45ef3da86394ecca37b3da4cb1fc0ce
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue May 6 16:48:36 2014 +0100

    get_unmet_load_hours added to analysis

commit 08279f338d2284e3cd7f58c671aeea591c7e0264
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue May 6 16:48:23 2014 +0100

    Bugfixes

commit 879681b0f5f65cb30093d934281665f542015f90
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue May 6 16:48:03 2014 +0100

    Unmet demand separate from supply again

commit 5f5b01bf51513d3fe428f613df27ed63bee6f133
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue May 6 11:01:40 2014 +0200

    Fixes to HDF solution storing and reading

commit 2b29433d703fc14d875d4370999e5206f9618dc8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon May 5 15:06:39 2014 +0200

    Save solution to HDF5 by default instead of to CSV

commit 1abf63a0772b285c513552f76b029e0b10fd4fba
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon May 5 09:17:09 2014 +0200

    Improved solution processing and saving

commit 1f96adce1b6bbaadd69a4f85cf250243169ab26a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 2 15:22:03 2014 +0200

    Added `areas_below_resolution` to analysis.py

commit 2f01b892a83d1c313a970a799c011ad8a1ceb73b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri May 2 14:38:27 2014 +0200

    Added group-based shares to solution processing

commit a3f3c8bacfe90984180d376b8401519a96948076
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu May 1 09:45:26 2014 +0200

    More fixes to time_tools

commit fc5a0bdb923fd914d7357e549401ec0e6ad6841c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 30 19:10:00 2014 +0200

    Vastly simplified time_tools

commit c8f08fff5e8b595207df6f612f8fe3c44124bee3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Apr 30 18:20:30 2014 +0200

    Bugfixes and debugging improvements

commit c1e6689da54ddcc045fc95136ac0cf5a5b74591a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 25 18:25:10 2014 +0200

    Tech parent sanity checking

commit 9501c5fcad35243787ec84f668d0d7f2ac2f3a8e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Apr 25 11:40:43 2014 +0200

    More analysis functionality

commit 1d3339a3ca99aebadd6389dbd3cca0384e934a2b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 24 15:46:33 2014 +0200

    Changes to availability (still buggy, disabled by default)

commit c08bad6c2605904ed4c0a1a30ec0f21fbb686aa1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 24 10:06:07 2014 +0200

    Bugfix; moved some functionality to analysis.py

commit a9b030899f8a1349165ef8badf75c6542f0d3579
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 24 09:57:05 2014 +0200

    Core and utils improvements

commit bb14c301268613a25495772cb3189974dfce4913
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Apr 15 11:18:53 2014 +0100

    Added availability parameter (a), simplified time_tools
    
    * Sacrificed speed for simplicity, will need to improve time_tools
      in the future

commit 3470f64239db3216d2ef89891025974d944708d9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Apr 14 16:32:57 2014 +0100

    Fixes to core and time_tools
    
    * Added r_unit option to differentiate between power and energy
    * Allow to specify debug.run_id option to override run_id
    * Fixed eff_ref bug
    * Ensure time_res_series correctly set on timestep adjustment
    * unmet_demand now child of 'supply' rather than 'defaults'

commit 5ee77c0fe1b955c2dfe05c5c18b4eca6be94bb34
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Apr 3 14:32:41 2014 +0100

    Minor core improvements

commit d62ecf0d6a7d7c2e65eb303b3fdad7dbcc9de30a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Mar 24 14:47:37 2014 +0000

    Bugfixes
    
    * storage techs
    * ramping constraint
    * time_res no longer a param

commit f43c2f6c3fc4abb56f6f167b32e84b1c6c9fcd42
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 18 14:56:50 2014 +0000

    Fixes to storage and transmission constraints

commit ea8c5b6bc21ddb75dd38941187688b6c74ec7494
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 18 10:51:41 2014 +0000

    0.2.0

commit 70e94d4a2d7ddb89ef142c00c2fc8b7cbcad3879
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 18 10:47:16 2014 +0000

    Iterative re-construction of parameters; other fixes

commit 3396f90cba5bb14975c6c60869a95c96bb85bd7f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Mar 16 08:13:51 2014 +0000

    Simplified constraints for speed

commit 827d4913cfd708ed8f0fc38d36a397b4fa3b1de9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 14 11:16:01 2014 +0000

    Fixed possible source of bugs in solver options

commit d7dfb646d30131a898d54eb2703bb2591bd903c8
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 7 13:11:36 2014 +0200

    Improved stack_plot

commit c48b151da7ad93c34d8d1d188c9f1939b928c7b5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Mar 7 13:11:09 2014 +0200

    Minor fixes

commit 90a40e483874fa005ca2fc6e5bcd01f7c62be905
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Mar 6 21:02:13 2014 +0200

    Fixed time subsetting bug in core

commit 2a489331b0653fb6341c776f2b41d03444f00a60
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 5 15:29:27 2014 +0200

    Fixed regression in AttrDict.to_yaml

commit 0f4138a259b0eab04de563017abf08b455346b42
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 5 15:05:49 2014 +0200

    Minor fixes

commit a93011af330239dca5dcbd42f4c3d269437ef430
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Mar 5 10:02:47 2014 +0200

    AttrDict fix, base constraint simplifications

commit e06dc22da825f07bf71795dd767ba3c67b71674f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 4 16:29:29 2014 +0200

    Better group_fraction

commit c4a9bb02752a4be51e8e1a665672240be67f8bd1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 4 10:43:01 2014 +0200

    Simplification

commit 71c84e7b54d044979945604f75b8e1246baffb8d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 4 10:04:19 2014 +0200

    More efficient get_var()

commit f133ce994dad9ed0ca88946d0abff3457976b121
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 4 09:10:17 2014 +0200

    Minor fixes and speedups

commit ebdcb5e3af2323644f3f597f5e71cbcfd8566cb0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Mar 4 09:09:41 2014 +0200

    Ability to define timestep options

commit 504bc44737dda71747d84f451715f52ec546d2e5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Mar 2 13:15:51 2014 +0200

    Bugfixes

commit 76dd1d128449166ea8c1e151c9915508bb3d4847
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Feb 28 14:20:19 2014 +0200

    Added group_fraction constraint; other fixes
    
    * Technology groups via inheritance chain, used for group_fraction
    * Improved iteration definitions
    * Small fixes to base constraints
    * Cleaned up defaults.yaml
    * Added `s_cap_max_force` option
    * Added `debug.delete_old_logs` option

commit 08bf519ebda91d79f23a8061d769fd8f184ce10f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Feb 27 22:09:39 2014 +0200

    Improved stackplot

commit ba993250cf7f35af778a464980a5036c0e3e3606
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Feb 27 21:58:16 2014 +0200

    Disabling os, can cause bugs

commit 10a2057baec560fd4b09d506ba776e16238653a5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Feb 25 15:22:15 2014 +0200

    Removed tech classes; added area_per_cap constraint
    
    * Former tech class functionality now handled in base constraints

commit d543db8fbfe1b50ffb6081c128ed72d4247e916e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Feb 25 11:28:43 2014 +0200

    Doc for system margin constraint

commit 0403e6ea55ff9934571687c3acf2714a199a7183
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Feb 25 11:17:00 2014 +0200

    Added system_margin constraint

commit 01f99cf1c7a1bf95279228a4c0f5c0893cede1d0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Feb 25 11:14:57 2014 +0200

    Minor fixes; passing all tests again

commit 2b3ca860dba1e4cb52f8e1bfc9656dddbb5df2d0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Feb 24 09:44:49 2014 +0200

    Various fixes, particularly to operational mode

commit c3eb6d35707c651aaa41bd8246794d21e6e42343
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Feb 10 18:58:53 2014 +0000

    Added stack_plot to utils

commit f0bb2a4a340517f0d2ab4f8b999506538fce9fd9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Feb 10 18:55:56 2014 +0000

    Improved parallel_tools

commit b352cc65ac16975697cf1cf92a3e0b60b3ceed8b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Feb 10 18:04:03 2014 +0000

    Major changes to time masks
    
    Breaks unit tests -- more changes and fixes to come

commit 96666e5006159fe1154e74f6dbf64d28841f82a5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Feb 7 09:08:42 2014 +0000

    Require pandas 0.13; minor compatibility fixes

commit 8b41ac430b8ffcc0e0905f1ad5710de68f5ca960
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jan 31 14:39:07 2014 +0000

    Changes to parallelizer and core config handling
    
    Core:
    * Now define either a single 'input.model' file or a list of files, no
      longer possible to define arbitrary 'input.' keys
    * Fix to output saving
    
    Parallelizer:
    * Fixed generate_iterations()
    * Convert numpy types to python types to prevent YAML writer choking
      (compat with newer pandas versions)
    * Don't save parallel key to individual run config files
    * Combine all nested config files into one 'model.yaml' file

commit cb20ca1c52076990cbf2a09dd8631c07535b1f13
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jan 31 10:36:42 2014 +0000

    Fix for failing locations test

commit fb3da463dfc4aa584e747cf6b9b57c41d46b710e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jan 31 09:59:52 2014 +0000

    Various fixes
    
    * Relative config paths as relative to run.yaml
    * Locations processing fixed (including order inside core)
    * Allow AttrDict to be shallow-copied via .copy()
    * Allow r_scale to be overridden per x

commit 41ea715225e08afc12dc39e72690854ec8c8d05f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jan 31 09:56:36 2014 +0000

    Added e_cap_max_scale option (undocumented)

commit cb2ec57612973cd86a2fadd77856138152aab6ff
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Jan 29 12:04:43 2014 +0000

    Capacity factor constraint fixes
    
    (oops.. this is actually non-linear)

commit 3f0c6d9524a9a9cce0f4bb2a9a5813848c2c9307
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 28 19:58:15 2014 +0000

    Minor doc changes

commit cf4f37e264c7ff8240a69f0debb9918c5d541182
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 28 19:57:44 2014 +0000

    Improvements to config handling:
    
    * 'import:' resolution now in AttrDict class
    * better locations management, including the ability to define settings
      for a location spread out over several definitions, e.g. ``1,2,3``
      followed by ``1`` and ``2``, etc.
    * Except for overriding defaults.yaml, an error is now raised if the
      configuration files attempt to define any setting twice (as no clear
      hierarchy was defined to resolve such conflicts)

commit 2ea934509f43a70c4112cb385d19700453813ee0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 28 15:27:00 2014 +0000

    pandas 0.13 compatibility; fix commit 550da05e22

commit 0330474f56e0c6daa66ed7965de74cb1e3c8da2c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Jan 28 14:29:54 2014 +0000

    Added cf_max constraint

commit 5d55a08adfdcdd3b37dd3c7db27c72ef3c47743e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat Jan 25 21:02:23 2014 +0000

    Added e_cap_max_force option

commit 1dd2ba21d623a63f6c2c73d712364b16542eafff
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Jan 24 11:50:08 2014 +0000

    Added om_fixed costs

commit 31d867a62a274071e3511d0b15ba167b7161e583
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Jan 6 15:04:39 2014 +0000

    Removed e_prod constraints, added es_prod_min

commit 564a062c7c3124f95757ca507f3a5ba3ba8c55d4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jan 2 15:26:18 2014 +0000

    Documentation updates: constraints and components

commit 969c40276c243584925b910442731a73e61247be
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jan 2 15:20:40 2014 +0000

    Minor constraint improvements

commit a865cdf73bdb30cf2ebc391ae6442d4d496bf257
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Jan 2 15:19:42 2014 +0000

    Apache 2.0 license

commit bcc44dcfc3c5d5267661093ca8362b11ba9c952e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 18 12:20:18 2013 +0100

    Cleaned up secondary resource and its doc

commit 6cc4dd60aa1963b19765f84a2c667fa6775bd4c1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 18 10:26:09 2013 +0100

    Removed (unused) _reduce_weighted_average from time

commit 414764f48ef46f9b6539b3b1fa2ba92cc437e9c2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Dec 18 10:25:10 2013 +0100

    Removed r_eff_ref, r_eff no longer a parameter, fixed e_eff_ref

commit 5769ff8f322208e585c96fba46f6360ad8a1b841
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 16 12:08:11 2013 +0100

    Implemented conversion technologies; cleaned up constraints

commit 06e1db1a5149f7aa1c982be5f67bc56d6b18a00e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sat Dec 14 18:56:17 2013 +0000

    Very barebones implementation of energy carriers

commit 82cb59f298a223cd354a46b61b507ff6122c7c68
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Dec 13 08:26:36 2013 +0000

    Generalize configuration file hierarchy, updated docs, minor other fixes

commit 4c6540bfce9697ad5bf633e0d40be8ab240df9ce
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 12 17:39:53 2013 +0000

    Added cost classes to allow for emissions accounting

commit 88d2ec311b829bd292d83e060aceba1bf359e8e7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 12 14:58:44 2013 +0000

    Cleared up terminology and added license
    
    * Both "nodes" and "locations" have a specific and different meaning
    * Most instances of "nodes" are now "locations"
    * Added GPLv3 license information
    * Updates to documentation

commit 2e366bdb7a21ccc9002532051b565123568e9434
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 10 18:04:31 2013 +0000

    Release 0.1.0

commit 3a0d290187ac171d324146e488150754d2d32acc
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 10 18:00:52 2013 +0000

    Minor AttrDict fixes and docstring adjustments

commit 53b3bd850134ce617bd93bc28ff31f05d97fea1d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 10 14:32:17 2013 +0000

    Ramping constraint for both directions; Doc additions

commit 5049149672afdff89c8222dae9ae131739426cf2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 10 13:30:37 2013 +0000

    Documentation updates

commit 69c73db761d2018d48ea876419fbe2ddb45c9729
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 10 12:04:21 2013 +0000

    Configuration extensions, ability to easily add constraints
    
    * Ability to add constraints via YAML
    * Reorganized constraints in subdirectory
    * Added ramping constraint as demo of optional constraint
    * Some fixes to how configuration is initialized in core

commit 12483ea8273780f547e300c92fde58d271b08e39
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 10 09:34:19 2013 +0000

    Improvements to time, some tests, some docs added

commit 36aa74ddf05b12ab36eb246c1cc219befd47b879
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 9 14:19:15 2013 +0000

    Improved masks: allow arbitrary adjustment patterns

commit 2d96f38288df0b368fb9da87b73753160b0e16d3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 9 13:36:18 2013 +0000

    Fixes to TimeSummarizer, re-added masking functionality

commit 12290d7aadeb619cbe5198cd6c132484941f81c5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 9 11:24:15 2013 +0000

    Working TimeSummarizer with reduced funtionality
    
    * working reduce_resolution method
    * no more weighted_average
    * removed mask, will be re-implemented separately

commit c5a000cd30e0af2eec60bfc25887ef2165d1bcb2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Dec 9 11:18:35 2013 +0000

    Fixed bug in model tests

commit b664d37b575b3fd6a49184ee257661c08c53038b
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 5 14:56:30 2013 +0000

    Updated documentation, bump version to 0.1.0-dev

commit bf54c44e6f6ffbe02f631026385596b155cdaaf2
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Dec 5 14:53:23 2013 +0000

    Various improvements to core and config
    
    * Allow to pass either path OR AttrDict to init Model()
    * Added additional override argument to init Model()
    * Allow to specify operation mode in YAML
    * Allow to specify solver in YAML
    * Improved utils and tests accordingly

commit d7413aa1190d48cfc103f177c236cb08e7ebaa4a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Dec 3 08:55:46 2013 +0000

    Improved example model

commit 76d6be2879e0e76177cea3209912c6b788779252
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Dec 1 15:03:51 2013 +0000

    Improved parallel runner

commit 539aaf18bbb6e964d54542270a9110fa97d6cb5f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Dec 1 15:03:11 2013 +0000

    Updated AttrDict: to_yaml() method and better set_key() error handling

commit 27d8e73feeee621bd5dc3834c4986c0de2eeb19f
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Dec 1 12:18:11 2013 +0000

    Added variable production resource test case

commit d8efc4cafb8866fe434407a86180b6827c548577
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Dec 1 11:57:12 2013 +0000

    Cleanup and fixes for model tests

commit 8deaf83b26965b8688f84a1404302e5d006891ac
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Dec 1 11:45:45 2013 +0000

    Added simple model with storage test case (+bugfixes)
    
    * Fixed a bug in get_var (need to extend unit test coverage to result getters!)

commit 7620bf25ccfa756ca11d761e476a5678d9ccd747
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Dec 1 10:26:31 2013 +0000

    Added several tests solving simple models

commit 6d506046e6dba1a9fd7ddb4d11b4a80df768327d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Dec 1 10:25:21 2013 +0000

    Fixed string bug in nodes, updated tests

commit cfe7c8b3a163f078ebaf89db15a716f2c34f38bf
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 29 09:55:46 2013 +0000

    More core tests and fixes

commit b9e9d360347dc2e9bb70f0ad10eb634948861cb5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 28 13:43:37 2013 +0000

    More core tests, including tests for get_timeres
    
    - Slightly modified get_timeres to allow verification of resolution across
      entire time series. NB: this verification fails if the time series skips
      some days, e.g. leap days!

commit 329becaebcd39198bbcd4c3f6992d8f67aaa929e
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 28 13:13:38 2013 +0000

    All techs now have r:0 by default; simple default model in module
    
    r:0 as default means that techs need to specify r:file or r:file=filename.csv
    explicitly, but also that the model no longer looks for inexistent files
    for technologies that exist somewhere in the model but not at a specific node

commit 20a1be77c6616f6847137f2fc16ea4bb81522e90
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 28 13:11:16 2013 +0000

    Slight change to how {{x}} placeholders work

commit e1f301d208415b736e0246890c30904cb6c14ded
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 27 18:43:26 2013 +0000

    Fixes to and some tests for core, minor other changes

commit 8c87a4d22c6f69e6320ffcbe47782e74a8e13abd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 27 10:00:16 2013 +0000

    Added tests for nodes

commit a5970be3038a33599a5dd74663cfc4d5bbb476a3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 27 08:35:42 2013 +0000

    Improvements to and tests for techs and transmission

commit 5f5d6c566a1cb6e5ba101111d2b54f2c157508ee
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 26 20:08:54 2013 +0000

    Improvements to utils and better test coverage

commit 495346628ffbfc6c4f560bd176b22a77c402d3b1
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 26 11:38:05 2013 +0000

    Restructured configuration files

commit 7a11b267c18eb607dd352654a09c8fcd0f9660f3
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 26 09:38:11 2013 +0000

    Cost calc fixes and minor changes to comments

commit 0986eb2403c72d73feb277aaad7af2f4779062cf
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Nov 25 12:13:16 2013 +0000

    Moved calliope_run script into parallel.py

commit 18d8d3e6894a730545fd179c5d258677bf59844d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 22 08:28:31 2013 +0000

    Let there be documentation

commit 7ee7d0b8de8456c4d7bcb5884ff23c9e3944d512
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 21 18:11:52 2013 +0000

    Symmetry constraint for transmission capacities

commit d43336f584018c9529b9bae33ac9a3b70a102fb9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 21 17:26:52 2013 +0000

    get_var returns 0 for unitialized values

commit caa9f1e0ab3e47bc24dd6ef0c4cd1ad10fc0fc3c
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 21 17:26:05 2013 +0000

    Added reference efficiencies, changes to balancing constraints

commit d0c641d712af2a10a46773d463fa88ae43985128
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 21 14:54:05 2013 +0000

    Fix typo

commit 983d2c7a9283d65c149128704f171819d0dd59ac
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 21 14:39:35 2013 +0000

    Fixed: use custom tech classes in initialize_techs

commit b18b2579916dc395eb87dbe6b0f09d45897a81cd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 21 13:57:37 2013 +0000

    Proper packaging allowing installation via pip

commit 4be514e63bc8e61f2a09b9d6606b8301ea6faea9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 21 11:00:09 2013 +0000

    Replaced model slack with unmet_demand technology

commit 959cbbe85dfe00108b4187e6610ee4106b20dd90
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 21 10:36:03 2013 +0000

    Balancing within parent nodes and transmission between nodes

commit 00ad59da48e903ab107e7f21859c2c417d7556f5
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 20 09:14:01 2013 +0000

    New add_constraint method

commit f35838f59a9e39fe9d313f3a26b923c23914eccd
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 19 16:32:38 2013 +0000

    Split e and es into e(s)_prod and e(s)_con as prep for transmission loss

commit 6faa7807ff548386a86a4ff5f506f3a1cb8f1d37
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 19 12:48:17 2013 +0000

    Improved nodes definition and data handling to allow per-node resource files

commit 85ba969f47b0c636b8eebf36cf977484fc604e25
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 19 08:47:45 2013 +0000

    Fixed solve_iterative

commit 4a6d8446c6e3bf50d5bc334746c7798b261b1bd0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Nov 18 13:00:31 2013 +0000

    Fix last commit: remove superfluous constraint overrides

commit 591e879a37fb02c3f0b04b8b8ac17c1a23e97eca
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Mon Nov 18 12:54:10 2013 +0000

    Major changes to settings and constraint overriding
    
    * Major changes to settings structure
    * No more aliases, instead, parent-child relationship between techs
    * 'techs_defaults' as a special parent to all techs
      that define 'parent=default'
    * Ability to override some constraints on a per-node basis

commit ea2941efff81acd6bfd414cef3e60a807a597225
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Nov 17 21:14:04 2013 +0000

    Core: fixed get_timeres() and process_/save_outputs()

commit 227360e7dc15845e0acff300d757dd516e9d533d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Nov 17 21:12:26 2013 +0000

    Improved node_energy_balance and node_constraints_operational

commit 6890df3087f7c7977a3d058a3340ddc4a0bbb93d
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Sun Nov 17 21:11:01 2013 +0000

    Re-added memoization to get_option()

commit e2ce0c7f8d80de5e6a45b3f8cfc943ba427d67c0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 15 22:21:07 2013 +0000

    Added tech aliases

commit 8ae802901401eb0f9877ca171cd8b92e528be694
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 15 22:19:43 2013 +0000

    Small fixes

commit dbb334c547bb00a72925d9c34624c236365a7a60
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 15 21:32:21 2013 +0000

    Better structured initialization; only instantiate used tech classes

commit be6d80b109c7c4919632811d6bcce4871e26a936
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 15 20:13:48 2013 +0000

    Read tech set (y) from nodes settings

commit 6f2d864790334241fe2f2d1f101dc0afd1413c9a
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 15 20:08:34 2013 +0000

    More versatile node selection

commit addbeb20ce4185220c45e5527f62dbed43cef0c4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 15 20:07:39 2013 +0000

    int keys converted to str in AttrDict initialization

commit 7a534b9df8ca198c696cd0d17e3e292e15b89908
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 15 19:33:10 2013 +0000

    Severe bugs fixed; allow demand resource and negative flows

commit 89b8d95d995c305c251e107927e88bad4dfb1764
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 15 13:50:59 2013 +0000

    Ability to restrict technologies to certain nodes

commit f304763271ee94e4492d46de055c659e2fe5adda
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Fri Nov 15 13:48:18 2013 +0000

    Minor fix to (currently ad-hoc) backup<>storage constraint

commit b41373af68ace748f65f7fb7f58f0f3211fb75a4
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 14 15:48:13 2013 +0000

    Nodes set (x) indices are now strings rather than ints

commit f6fa1d63dc1abf892981304462100826975d83e7
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Thu Nov 14 15:42:58 2013 +0000

    Minor changes to how technologies set is defined and subset

commit 23c940cdb69b54e7ee63d0ebb62c9a7f1abe65ac
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 13 18:26:31 2013 +0000

    Updated .gitignore; removed Makefile

commit 1b28f692144e2fac4239f3f82d6528d33dcd6e76
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 13 18:22:08 2013 +0000

    Working multi-technology version; various fixes

commit a2978fa040b59802c73a6a1368954ce622449da0
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 13 11:25:00 2013 +0000

    Result getters updated

commit 4591882221c412ec43e9ef7093cd3890c1ad0817
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Wed Nov 13 11:22:42 2013 +0000

    Clean up constraints

commit 4887b812fd55d3191ee45a6c68a4e1c47a6638f9
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 12 20:01:53 2013 +0000

    Various fixes, model runs successfully in planning mode

commit 5d823eb6266f4e6b2b12c8c610a62b5a3aac9e26
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 12 18:10:18 2013 +0000

    Removed memoization for now

commit 38f80a1f1af49422fe6fb657958e51dd81991b81
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 12 18:09:34 2013 +0000

    Updated README

commit 9950d4872c98e32ea09d5d883de543cff04dfb31
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 12 16:58:27 2013 +0000

    Added barebones README

commit f0dff09e1c5ff57ae153ac0a779af98d2c6166dc
Author: Stefan Pfenninger <stefan@pfenninger.org>
Date:   Tue Nov 12 16:51:02 2013 +0000

    Calliope begins
