#!/bin/bash


ENV="ocean" # "ground" "desert" "wetland" "clay" "ice" "snow" "shallow_water" "ocean"
ANIMAL="Fish" # "BabySeal" "Caterpillar" "Panda" "Fish" "Fish_2" "GreatWhiteShark" "Orca"
SEED=100
OUT_DIR="./local/archetype_embodiment/${ENV}-${ANIMAL}-${SEED}"
CONTROLLER_CKPT="./local/diffsim/${ENV}-${ANIMAL}-${SEED}/ckpt/controller/iter_0020.ckpt"

args=(
    --out-dir $OUT_DIR
    --objective "embody_archetype"
    --env "aquatic_environment" # NOTE: uncomment this if using ocean environment
    --env-config-file ${ENV}.yaml
    # --save-every-iter 1
    --render-every-iter 1
    --n-iters 31
    --n-frames 200
    --loss-types FinalStepCoMLoss
    --designer-type annotated_pcd
    --annotated-pcd-path ./softzoo/assets/meshes/pcd/${ANIMAL}.pcd
    --annotated-pcd-passive-geometry-mul 0.2
    --torch-seed $SEED
    # controller args
    --load-controller $CONTROLLER_CKPT
    --optimize-controller
    --controller-lr 0.1
    --actuation-omega 20. 100.
)
python -m algorithms.diffsim.run_archetype_embodiment "${args[@]}"
