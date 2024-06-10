#!/bin/bash
#this script was given courtousy of annemarie rickmann -- lead author of v2cc.
export FREESURFER_HOME=/mnt/nas/Software/Freesurfer70
source $FREESURFER_HOME/SetUpFreeSurfer.sh


export SUBJECTS_DIR=/mnt/nas/Data_Neuro/ADNI_FS72
export SAVE_DIR=/mnt/nas/Data_Neuro/ADNI_CSR/FS72/fsaverage_remapped


while read sub ; do
  echo $sub
    mkdir -p $SAVE_DIR/$sub
    mkdir $SAVE_DIR/$sub/mri
    mkdir $SAVE_DIR/$sub/surf
    mkdir $SAVE_DIR/$sub/label
    cp $SUBJECTS_DIR/$sub/mri/orig.mgz $SAVE_DIR/$sub/mri/orig.mgz
    for hemi in lh rh ; do
      for surf in white pial ; do

        # resample surface coordinates
        mri_surf2surf --s $sub --hemi $hemi --sval-xyz $surf --trgsubject fsaverage --tval $SAVE_DIR/$sub/surf/$hemi.$surf --tval-xyz $SAVE_DIR/$sub/mri/orig.mgz &
        
        # thickness
        mri_surf2surf --hemi $hemi --srcsubject $sub --srcsurfval thickness --src_type curv --trgsubject fsaverage --tval $SAVE_DIR/$sub/surf/$hemi.thickness --trg_type curv &

        # curvature

        mri_surf2surf --hemi $hemi --srcsubject $sub --srcsurfval curv --src_type curv --trgsubject fsaverage --tval $SAVE_DIR/$sub/surf/$hemi.curv --trg_type curv &

        # annot DKT
        mri_surf2surf --srcsubject $sub --trgsubject fsaverage --hemi $hemi --sval-annot $SUBJECTS_DIR/$sub/label/$hemi.aparc.DKTatlas.annot --tval $SAVE_DIR/$sub/label/$hemi.aparc.DKTatlas.annot &

        # annot Destrieux

        mri_surf2surf --srcsubject $sub --trgsubject fsaverage --hemi $hemi --sval-annot $SUBJECTS_DIR/$sub/label/$hemi.aparc.a2009s.annot --tval $SAVE_DIR/$sub/label/$hemi.aparc.a2009s.annot
      done
    done
  sleep 3 

done <  /mnt/nas/Data_Neuro/ADNI_FS72/FS72/all_subjects.txt
