import os
import glob


def main():
    print(f"Running test evals ...")
    checkpoint_pattern = 'ckpt_100000'
    run_count = 0
    eval_count = 0
    for game_dir in glob.glob(os.path.join('logs/runs/dreamer_v3', '*')):
        for run_dir in glob.glob(os.path.join(game_dir, '*')):
            run_name = run_dir.split('/')[-1]
            # if run_dir start with a number lesser than 2024-02-07 continue, 
            # early runs used a different ocatari wrapper with x,y,w,h instead of just x,y
            if run_name < '2024-02-07':
                print("skipping", run_name, "because it is an old run with different octari wrapper")
                continue
            for version_dir in glob.glob(os.path.join(run_dir, 'version_*')):
                # check if checkpoint subdirectory exists
                if not os.path.exists(os.path.join(version_dir, 'checkpoint')):
                    continue # no checkpoint directory found
                # Get checkpoint file path by looking for a file in the checkpoint directory that starts with 'ckpt_100000'
                checkpoint_path = glob.glob(os.path.join(version_dir, 'checkpoint', f'{checkpoint_pattern}*'))
                if len(checkpoint_path) == 0:
                    continue # no checkpoint found in the checkpoint directory
                checkpoint_path = checkpoint_path[0] # should be only one file
                # check if the evaluation directory exists
                eval_dir = os.path.join(version_dir, 'evaluation')
                test_evals_to_run = 0
                if not os.path.exists(eval_dir):
                    # crete the evaluation directory
                    os.makedirs(eval_dir)
                # check that version_N for N = 0,1,2,3,4 exists within the evaluation dir, if not add to the list
                for i in range(0, 5):
                    version_dir = os.path.join(eval_dir, f'version_{i}')
                    if not os.path.exists(version_dir):
                        test_evals_to_run += 1
                if test_evals_to_run == 0:
                    print(f"Skipping {run_name} because all 5 test evals have been run")
                    continue

                print(f"Running {test_evals_to_run} test evals for {checkpoint_path}")
                # run the evaluation
                for i in range(test_evals_to_run):
                    os.system(f"python sheeprl_eval.py checkpoint_path={checkpoint_path}")
                run_count +=1
                eval_count += test_evals_to_run
                
    
    print("Done",eval_count ,"test evals for", run_count, "experiments")


if __name__ == "__main__":
    main()