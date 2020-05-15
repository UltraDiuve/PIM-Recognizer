echo off
cd C:\Users\pmasse\Pyprojects\PIM-Recognizer\report\notebooks
echo Lancement 1/6
jupyter nbconvert --to pdf "Analyse donnees du PIM.ipynb"
echo Lancement 2/6
jupyter nbconvert --to pdf "Analyse quantitative.ipynb"
echo Lancement 3/6
jupyter nbconvert --to pdf ground_truth_constitution.ipynb
echo Lancement 4/6
jupyter nbconvert --to pdf gt_based_model.ipynb
echo Lancement 5/6
jupyter nbconvert --to pdf open_model.ipynb
echo Lancement 6/6
jupyter nbconvert --to pdf Performance_measurement.ipynb
echo Done
