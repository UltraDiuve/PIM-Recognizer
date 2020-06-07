echo off
cd C:\Users\pmasse\Pyprojects\PIM-Recognizer\report\notebooks
echo Lancement 1/8
jupyter nbconvert --to pdf "Analyse donnees du PIM.ipynb"
echo Lancement 2/8
jupyter nbconvert --to pdf "Analyse quantitative.ipynb"
echo Lancement 3/8
jupyter nbconvert --to pdf ground_truth_constitution.ipynb
echo Lancement 4/8
jupyter nbconvert --to pdf gt_based_model.ipynb
echo Lancement 5/8
jupyter nbconvert --to pdf open_model.ipynb
echo Lancement 6/8
jupyter nbconvert --to pdf Performance_measurement.ipynb
echo Lancement 7/8
jupyter nbconvert --to pdf model_tuning.ipynb
echo Lancement 8/8
jupyter nbconvert --to pdf Text_analysis.ipynb
echo Done