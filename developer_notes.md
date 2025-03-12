# Developer notes

Developers -- please add significant changes and notes for future development to this document :)

TODO: we really really need tests! For both `app.py` (callbacks) and also for `measles_single_population.py` as well! Especially with all the combinations and edge cases, we cannot rely on manual testing alone. 

TODO: we need a script to assess validity and format of new states' datasets!

## State Vaccination Data Status
- `TX_MMR_vax_rate.csv` 
- `NC_MMR_vax_rate.csv` -- new dataset without duplicates 03/11/2025 

### Fixing default values bug -- 03/12/2025 LP
- **Problem:** Initially we wanted to 
- **Impact:**
- **Fix:**
- **Technical details:**

## Tues Mar 7 - Mon Mar 11
All following changes documented and implemented by LP

### Streamlining user inputs and parameters
- **Problem:** Formats are not consistent between user inputs and the actual parameters (in a parameters `dict`) used in `measles_single_population.py`. Conversion is needed, but this conversion does not occur in an obvious place. For example, the user inputs vaccination rate as a percent (e.g. 95) but in the actual model parameters, this vaccination rate is a decimal (in a list, corresponding to arbitrary numbers of subpopulations, but recall we only use 1 subpopulation for now).
- **Impact:** Confusing units and a recipe for computation mistakes later.
- **Fix:** Create a callback that converts user input (via selectors) to proper model parameters format. 
- **Technical details:** Store `dashboard_params` as data in a `dcc.Store` object included in `app.layout`.

### Adding additional states' vaccination data
- **Problem:** Code is hard-coded for Texas's data. 
- **Impact:** Cannot easily add more states. Appending new state data to Texas's CSV is not a good idea for several reasons. First, the dataset becomes less manageable and modular -- it is hard to update an individual state's dataset or error check. Second, the dropdowns would not make much sense -- North Carolina counties, for instance, would show up even though the state dropdown is set to Texas.
- **Fix:** Generalize code to allow multiple states (note: non-trivial -- amending the code to allow 2 states is just as complicated as amending the code to allow N states). 
- **Technical details:** Create a `state_to_df_map` to manage datasets. Create callback that updates county selector based on state input. Lots of small technical fixes to the callbacks and selectors' default values (currently default values for state/county/school are set to a Texas school). 

### Input guardrails (very important)
- **Problem:** We need to have initial infected <= (1 - vaccination rate as decimal) * school enrollment. In the UI, users can enter combinations of school enrollment, initial infected, and vaccination rate that didn't make sense. Thank you Emily Javan for identifying this issue. 
- **Impact:** Nonsensical results and also non-transparent / confusing stuff for the user.
- **Fix:** Every time ANY of the three inputs (initial infected, vaccination rate, school enrollment) is changed, check the above condition. If the condition is satisfied, simulate as usual. If the condition is not satisfied, IMMEDIATELY return a red error message in the inputs panel. Also blank the results and the graph (set it to the default graph) so the user does not get confused and associate the wrong inputs with the results (this was a good point from Lauren).
- **Technical detail:** Callbacks are tough! Beware of callback loops/dependencies that break the code. (Iterated with Lauren for a bit with this.) We need `check_inputs_validity` callback to output TWO things: `Output('inputs_are_valid', 'data')` and `Output('warning_str', 'children')` -- `inputs_are_valid` is a `dcc.Store` object (that stores data) that is included in `app.layout`. `warning_str` populates the warning message (deep somewhere in the dynamic graphics code).
- **Important note:** Even if `measles_single_population.py` handles this internally, this is not sufficient -- and actually might even be really bad in the absence of UI input guardrails. Lauren pointed out that we do NOT want to change numbers in the backend (secretly) without alerting the user! That is not transparent. 

### `app.py`
- **Problem:** (1) Workflow was inefficient because we would make changes to the code and then Gladys would manually update them in a different github. (2) `app.py` got extremely long and complicated.
- **Impact:** (1) When running `app.py`, we would not see the delicious dashboard that Gladys made -- we saw a preliminary markup, and we were also out-of-sync with the most up-to-date UI/text changes. (2) Took a long time to find where to make small edits and figure out where everything was. Made reading/testing/etc... difficult.
- **Fix:** (1) We are doing everything in a centralized github: https://github.com/TACC/measles-dashboard -- developers should make branches and issue pull requests! (2) `app.py` has been separated into multiple files separated by function. We have clearly separated static graphics (e.g. text that does not change based on user input and simulation replications) and dynamic graphics (that do have these dependencies). HTML formatting and styling has been separated out and saved as reusable dictionaries to reduce clutter and redundant code. 
- **Technical detail:** Now `app.py` is much easier to read -- it only contains callbacks and the `app.layout` shenanigans -- large portions have been tucked away as functions for readability (e.g. we have stuff like `results_header()` rather than a mega chunk of HTML text). In the future it may make sense to also put the callbacks in a separate file. 

### Incidence
- **Request:** Wanted ability to track incidence -- new cases -- Lauren specifically asked for the daily number going from `S` to `E` each day. 
- **Addition:** Added capability to track incidence (essentially we track "transition variables") and store them just like compartments. Added graphing capability too. 
- **Technical details:** Please keep this in mind for discretization -- for statistics such as total infections, it is sufficient to just look at the end of the day (or in general, a single timepoint within a day) to get the discretized value. For statistics such as incidence, to get the total new infections in a day, we cannot look at a given timepoint in the day -- we have to sum across all timepoints! To make sure we properly compute the total new infections during that day. 
- **Note:** We do not currently use incidence in the dashboard, but we do have the capability to switch. We also may want to turn this off if we are not using this, to speed up the simulation execution time (because tracking and storing extra quantities takes time).

### Reproducible random numbers
- **Problem:** Refreshing / re-running simulation with same parameters gives different results.
- **Impact:** (thank you Kelly Gaither and Becky Wilson for pointing out) confusing and could lead to model distrust. Also for the future, might make results caching more difficult.
- **Fix:** the RNG starting seeds for both the RNG for selecting 20 out of 200 simulation replications randomly (for graph generation) and for getting random transitions (for the replications themselves) were a function of entropy and time -- this was not reproducible. Replaced each RNG to be "fixed" RNG from specific starting seed. 
- **Technical detail 1:** Instantiating a new `np.random.Generator` object (via jumping -- e.g. see Mersenne Twister generator and `jumped` method) for each simulation replication is very very slow. Instead, we'll just use the same generator at the beginning of every set of simulation replications. This solves this reproducibility problem.
- **Technical detail 2:** We don't think we ultimately encountered this issue, but based on research it is also possible to get weird non-reproducible random numbers due to multiprocessing and server issues! So, if we have future reproducibility issues we may want to look out for this.      

### Small stuff
- Lauren wanted the "default graph" to have sensible axes and look similar to the graph generated by running simulations -- the previous default graph had negative numbers on the axes and unlabeled axes. Note that is actually crazy difficult (impossible?!) to override the default formatting when passing an empty dataset to a graph for this `plotly` method -- negative numbers on the axes appear inevitable -- so we suppressed the tickmark labels to get rid of the negative numbers. 
- Some small bugs with running `measles_single_population.py` directly because the parameters were not all there -- fixed. 
- Minor variable renaming to make certain quantities more clear (based on discussions).
- Minor refactoring of functions. Big example is that for `update_graph` callback, we have created separate helper functions. 
- (Apologies, having trouble remembering specifics for this one... but thank you Emily Javan for identifying this issue.) For the parameters school enrollment = 500, initially infected = 25, vaccination rate = 95% --> meaning 475 students are in "R" (immune) -- this means that S and E are 0 -- and there is no one left to be infected. The probability of 20 new additional cases should be 0%. However, the dashboard was showing 100% and 24 for likelihood of 20+ additional cases and outbreak size given 20+ additional cases, respectively. Fixed. 