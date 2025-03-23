## Folder: `iMSMS_dataset`
This folder contains Excel files with multiple sheets containing iMSMS data.

## File: `iMSMS_Correlation_heatmap.py`
This file contains Python code to generate correlation heatmaps from iMSMS data.

## File: `iMSMS_Mann-Whitney test.py`
This file contains Python code to generate the results of the Mann-Whitney test from iMSMS data.

## Folder: `iMSMS_emperor_host_static_html`
This folder contains example for final version of static individual module of emperor with iMSMS data which can be hosted in github pages.

## File: `iMSMS_emperor.py`
This file contains a Python script to generate a static Emperor file from iMSMS data.

After executing the script, the following files are generated:
* `iMSMS_generated_ordination_data.txt`
* `iMSMS-emperor.html`

### Steps:
1. **Create a new folder**:
   - Once the files are generated, create a separate folder and copy only the `iMSMS-emperor.html` file into that folder.

2. **Copy the required assets**:
   - Even though the HTML file is copied to the folder, it still requires other assets like CSS, JS, and other dependencies that are typically hosted in the Emperor folder.
   - We need to render the HTML independently as a separate module.

3. **Copy assets from Emperor**:
   - As demonstrated in the `iMSMS_emperor_host_static_html` folder, copy all other required assets from the Emperor directory.

4. **Modify asset paths in `iMSMS-emperor.html`**:
   - After copying the necessary assets, modify the asset paths in the `iMSMS-emperor.html` file. Specifically, change the absolute path of the assets:

```
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/emperor/support_files//vendor/css/jquery-ui.min.css
```

to

```
vendor/css/jquery-ui.min.css
```



