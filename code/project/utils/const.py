import os
from pathlib import Path


# ADNI data path
if os.environ.get("HOME") == "/Users/a16446":
    ACE_FILE = Path(
        "/Users/a16446/Documents/GitHub/BrainGenePathway/data/ASD/final_ACE_KEGG_pathway_with_all_genes_img_4_features_p_threshold_0.1_effect_size_LD_50kb.csv"
    )
    ADNI_FILE = Path(
        "/Users/a16446/Documents/GitHub/BrainGenePathway/data/ADNI/final_AD_KEGG_pathway_with_all_genes_img_p_threshold_0.1_effect_size_LD_50kb.csv"
    )
    CROSS_VAL_INDEX_ACE = Path(
        "/Users/a16446/Documents/GitHub/BrainGenePathway/data/ASD/10_10_cross_fold_val_index.pkl"
    )
    CROSS_VAL_INDEX_ADNI = Path(
        "/Users/a16446/Documents/GitHub/BrainGenePathway/data/ADNI/10_10_cross_fold_val_index.pkl"
    )
    RESULT_FOLDER = Path(
        "/Users/a16446/Documents/GitHub/BrainGenePathway/results/results"
    ).resolve()
elif os.environ.get("HOME") == "/home/mnd2vy":
    ACE_FILE = Path(
        "/data/ics328/ivy-hip-ubdt/JueqiWang/data/ASD/final_ACE_KEGG_pathway_with_all_genes_img_4_features_p_threshold_0.1_effect_size_LD_50kb.csv"
    )
    ADNI_FILE = Path(
        "/data/ics328/ivy-hip-ubdt/JueqiWang/data/ADNI/final_AD_KEGG_pathway_with_all_genes_img_p_threshold_0.1_effect_size_LD_50kb.csv"
    )
    CROSS_VAL_INDEX_ACE = Path(
        "/data/ics328/ivy-hip-ubdt/JueqiWang/data/ASD/10_10_cross_fold_val_index.pkl"
    )
    CROSS_VAL_INDEX_ADNI = Path(
        "/data/ics328/ivy-hip-ubdt/JueqiWang/data/ADNI/10_10_cross_fold_val_index.pkl"
    )
elif os.environ.get("HOME") == "/usr3/graduate/jueqiw":
    ACE_FILE = Path(
        "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ACE/final_ACE_KEGG_pathway_with_all_genes_img_4_features_p_threshold_0.1_effect_size_LD_50kb.csv"
    )
    ADNI_FILE = Path(
        "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ADNI/Gene/final_AD_KEGG_pathway_with_all_genes_img_p_threshold_0.1_effect_size_LD_50kb.csv"
    )
    CROSS_VAL_INDEX_ACE = Path(
        "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ACE/10_10_cross_fold_val_index.pkl"
    )
    CROSS_VAL_INDEX_ADNI = Path(
        "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ADNI/10_10_cross_fold_val_index.pkl"
    )
    RESULT_FOLDER = Path(
        "/projectnb/ace-ig/jueqiw/experiment/BrainGenePathway/results"
    ).resolve()

DATA_FOLDER = Path("/projectnb/ace-ig/jueqiw/dataset/ADNI").resolve()

TADPOLE_FOLDER = DATA_FOLDER / "tadpole"
P_VALUE_FOLDER = DATA_FOLDER / "genetics_data" / "p_value"
SNP_FOLDER = (
    DATA_FOLDER
    / "genetics_data"
    / "ADNI_Test_Data"
    / "ImputedGenotypes"
    / "plink_preprocess"
)

# ACE data path
# ACE_GENO_FILE = "/projectnb/ace-ig/dataset/ACE/genetics/prs_analysis/data/ACE/GMIND/selected_geno_pheno.csv"
ACE_GENO_FILE = "/projectnb/ace-ig/jueqiw/dataset/ACE/genetics/prs_analysis/data/ACE/GMIND/total_geno.csv"
ACE_IMG_FILE_DESTRIEUX = "/projectnb/ace-ig/jueqiw/dataset/ACE/mri/freesurfer/group_stats/sMRI_destrieux_cortical_thickness_average.csv"
ACE_IMG_FILE_BRAINNETOME = "/projectnb/ace-ig/jueqiw/dataset/ACE/mri/freesurfer/group_stats/ACE_img_Brainnetome.csv"
ACE_IMG_GENO_FOLDER = Path("/project/ace-ig/jueqiw/data/ACE/joint")

# +-----------------+
# | ACE paired path |
# +-----------------+
ACE_PAIRED_FOLDER = Path(
    "/projectnb/ace-ig/jueqiw/dataset/ACE/genetics/prs_analysis/data/GMIND_new/pair_train_val_test"
)

# +------------+
# | SSC Folder |
# +------------+
SSC_FOLDER = Path("/project/ace-ig/jueqiw/data/SSC/")
ACE_IMG_GENE_INNER = Path(
    "/projectnb/ace-ig/jueqiw/dataset/ACE/genetics/prs_analysis/data/ACE/GMIND/ACE_img_Brainnetome_geno_inner.csv"
)
