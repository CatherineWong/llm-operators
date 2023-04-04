### LLM-Operators
Learning planning domain models from natural language and grounding.

### ALFRED experiment quickstart. This sets up the repository to run experiments with the ALFRED dataset.
The lab notebook for these experiments is located at `alfred_experiments_README.md`.
The following setup has been tested on an M1 Mac.
1. Install GIT LFS if you don't already have it ([see this help documentation.](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage))
2. *Download the ALFRED PDDL files*. 
```
git clone git@github.com:CatherineWong/llm_operators_datasets.git 
unzip llm_operators_datasets/alfred_linearized_pddl.zip
mv -R llm_operators_datasets/alfred_linearized_pddl llm_operators/data/dataset/alfred_linearized_pddl
```
- Developer notes: this is a preprocessed dataset of ALFRED problem files that contains the PDDL paths referenced in `dataset/alfred-linearized-100-NLgoals-operators.json`. To prepare this from scratch, use the raw PDDL files hosted [here](https://drive.google.com/file/d/1sg8v1hf40Eu1K7hLGZ_LP5I-9N4zwLCU/view?usp=sharing), and is originally copied from the MIT internal version at `/data/vision/torralba/datasets/ALFRED/data/full_2.1.0/`. We extract this to `dataset/alfred_pddl`; you should see three internal folders (train, valid_seen, valid_unseen). This provides the PDDL paths referenced in `dataset/alfred-NLgoal-operators.json`.
   - Prepare the ALFRED PDDL files. We modify the ALFRED domain to support simple fast downward planning. Run `prepare_alfred_pddl.py` to do so.
2. *Install the ALFRED Git submodules*. 
```
git submodule update --init --recursive`
```
- This installs the following two submodules:
   - `pddlgym_planners`. This contains the fast-downward task planner that we use in llm_operators/task_planner.py
   - `alfred`. This is a fork of the main ALFRED repository (https://github.com/jiahai-feng/alfred) that we have updated to support task and motion planning from PDDL with custom operators.

3. *Install the pretrained motion planner weights*. These are also located in the llm_operators_datasets. 
```
unzip llm_operators_datasets/t5-small-finetuned-alfred-long.zip
mv -R llm_operators_datasets/t5-small-finetuned-alfred-long llm_operators/alfred/t5-small-finetuned-alfred-long
``` 
- Developer notes: The Alfred motion planner uses a T5 module that been pretrained to support low-level planning towards the predicates in ALFRED. 

4. *Add an OpenAI environment key.* If you do not have one, you can add in any dummy key. This will just not be able to run the OPENAI portions from scratch. Edit `.bash_profile` to contain `export OPENAI_API_KEY=<OPEN_AI_KEY>` and run `source .bash_profile`.

5. *Create the ALFRED Python environment.* We assume you have conda. We use the following conda setup, tested on M1:
```
conda create --name llm-operators-38 python=3.8
pip install -r alfred_requirements.txt
```

6. *Modify the AI2Thor installation.* This is the jankiest part - the AI2Thor installation required for ALFRED uses a deprecated numpy version. To edit this, you should locate the local installation of AI2Thor within the conda environment. Start your environment, and run:
```
python # This opens an interactive terminal.
> import ai2thor
> print(ai2thor.__file__)
# This should output a location like /Users/<USER>/opt/anaconda3/envs/llm-operators-38/lib/python3.8/site-packages/ai2thor/__init__.py
```
Now modify the `server.py` definition in AI2Thor, using whatever code editor you like. For instance, in vim:
```
vim /Users/<USER>/opt/anaconda3/envs/llm-operators-38/lib/python3.8/site-packages/ai2thor/server.py

# Now, in vim, change the line that says 
   return np.asscalar(obj)
# You should change this line to read:
   return obj.item()
```

6. *Run a demo command to test the task and motion planner.*  (Last modified: 4-4-2023). This is under development. However, you can run the following command to test that the ALFRED-specific portions of this are working.
```
python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean_38354 --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_ground_truth_operators --debug_ground_truth_goals --assume_alfred_teleportation
```
- You should see this install the FastDownward (FD) planner inside `pddlgym_planners`. If this fails, see the `pddlgym_planners` repository ([here](https://github.com/ronuchit/pddlgym_planners.git)).
- You should see this open a Unity window in which motion planning succeeds.
- ==> Re-running this will write over the `experiment_name` directory. If you run this command locally, consider adding a timestamp (eg. --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean_3_3`.
--------------------------------------------
##### Adding in new domains. 
The following describes how we add in the ALFRED domain, which comprises a ground truth PDDL domain file, a set of individual PDDL tasks and NL annotations, and a motion planner.
1. Registering a PDDL domain: to register a new PDDL domain (which you can then specify using the `--pddl_domain_name` flag for `main.py`), you should register a new PDDL domain file loader, like the ALFRED example [here](https://github.com/CatherineWong/llm-operators/blob/main/datasets.py#L201), which initializes a new [Domain](https://github.com/CatherineWong/llm-operators/blob/main/pddl.py#L14) object. Our example also optionally implements a `operator_canonicalization` and `codex_types` attribute that is only used to construct Codex prompts, and is probably not necessary for new, non-ALFRED domains. In general, this PDDL file should contain both a set of ground truth operators and all of the predicates you want to plan over in FD and prompt Codex with.
2. Registering a dataset of new tasks: register a new dataset loader (which you can then specify using the `--dataset_name` flag for `main.py`) , like the example [here](https://github.com/CatherineWong/llm-operators/blob/main/datasets.py#L447). This should load both a set of PDDL problem files (the ALFRED ones are downloaded separately, see below, ALFRED PDDL dataset) and NL annotations, and creates a {<SPLIT_NAME> : {task_id : [Problem](https://github.com/CatherineWong/llm-operators/blob/main/datasets.py#L14)}} structure for loading train/test splits of problems.
These two steps should be enough to get the basic task planning + Codex portion running. To then add a domain-specific motion planner,
3. Adding a motion planner: you'd add in another motion planner [here](https://github.com/CatherineWong/llm-operators/blob/main/motion_planner.py#L30).


