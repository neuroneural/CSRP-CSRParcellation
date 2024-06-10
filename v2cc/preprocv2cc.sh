#!/bin/bash
# This script was given courtesy of Anne-Marie Rickmann -- lead author of v2cc.

#set up variables in slurm script that calls and passes single subject instead of for loop
sub=$1
echo $sub
mkdir -p $SAVE_DIR/$sub
mkdir $SAVE_DIR/$sub/mri
mkdir $SAVE_DIR/$sub/surf
mkdir $SAVE_DIR/$sub/label
cp $SUBJECTS_DIR/$sub/mri/orig.mgz $SAVE_DIR/$sub/mri/orig.mgz
for hemi in lh rh ; do
for surf in white pial ; do

    mri_surf2surf --s $sub --hemi $hemi --sval-xyz $surf --trgsubject fsaverage --tval $SAVE_DIR/$sub/surf/$hemi.$surf --tval-xyz $SAVE_DIR/$sub/mri/orig.mgz &
    
    # Thickness
    mri_surf2surf --hemi $hemi --srcsubject $sub --srcsurfval thickness --src_type curv --trgsubject fsaverage --tval $SAVE_DIR/$sub/surf/$hemi.thickness --trg_type curv &

    # Curvature
    mri_surf2surf --hemi $hemi --srcsubject $sub --srcsurfval curv --src_type curv --trgsubject fsaverage --tval $SAVE_DIR/$sub/surf/$hemi.curv --trg_type curv &

    # Annot DKT
    mri_surf2surf --srcsubject $sub --trgsubject fsaverage --hemi $hemi --sval-annot $SUBJECTS_DIR/$sub/label/$hemi.aparc.DKTatlas40.annot --tval $SAVE_DIR/$sub/label/$hemi.aparc.DKTatlas40.annot &

    # Annot Destrieux
    mri_surf2surf --srcsubject $sub --trgsubject fsaverage --hemi $hemi --sval-annot $SUBJECTS_DIR/$sub/label/$hemi.aparc.a2009s.annot --tval $SAVE_DIR/$sub/label/$hemi.aparc.a2009s.annot
done
done
