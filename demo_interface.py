import streamlit as st
import tifffile
import pydicom
from scipy.ndimage import zoom
import torch
from core.models.dani_model import dani_model
import numpy as np
from PIL import Image
import base64
import time


# Funzione per convertire un'immagine in base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dati di esempio predefiniti
esempi = {
    "FRO -> LAT": {'FRO': 'FtoL.png', 'LAT': 'LfromF.png'},
    "FRO -> REP": {'FRO': '31d9847f-987fcf63-704f7496-d2b21eb8-63cd973e.tiff', 'REP': 'Small bilateral pleural effusions, left greater than right.'},
    "FRO -> LAT + REP": {'FRO': '81bca127-0c416084-67f8033c-ecb26476-6d1ecf60.tiff', 'LAT': 'd52a0c5c-bb7104b0-b1d821a5-959984c3-33c04ccb.tiff', 'REP': 'No acute intrathoracic process. Heart Size is normal. Lungs are clear. No pneumothorax'},
    "LAT -> FRO": {'LAT': 'LtoF.png', 'FRO': 'FfromL.png'},
    "LAT -> REP": {'LAT': 'd52a0c5c-bb7104b0-b1d821a5-959984c3-33c04ccb.tiff', 'REP': 'no acute cardiopulmonary process. if concern for injury persists, a dedicated rib series with markers would be necessary to ensure no rib fractures.'},
    "LAT -> FRO + REP": {'LAT': 'reald52a0c5c-bb7104b0-b1d821a5-959984c3-33c04ccb.tiff', 'FRO': 'ab37274f-b4c1fc04-e2ff24b4-4a130ba3-cd167968.tiff', 'REP': 'No acute intrathoracic process. If there is strong concern for rib fracture, a dedicated rib series may be performed.'},
    "REP -> FRO": {'REP': 'Left lung opacification which may reflect pneumonia superimposed on metastatic disease.', 'FRO': '02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.tiff'},
    "REP -> LAT": {'REP': 'Bilateral pleural effusions, cardiomegaly and mild edema suggest fluid overload.', 'LAT': '489faba7-a9dc5f1d-fd7241d6-9638d855-eaa952b1.tiff'},
    "REP -> FRO + LAT": {'REP': 'No acute intrathoracic process. The lungs are clean and heart is normal size.', 'FRO': 'f27ba7cd-44486c2e-29f3e890-f2b9f94e-84110448.tiff', 'LAT': 'b20c9570-de77944a-b8604ba0-73305a7b-d608a72b.tiff'},
    "FRO + LAT -> REP": {'FRO': '95856dd1-5878b5b1-9c104817-760c0122-6187946f.tiff', 'LAT': '3723d912-71940d69-4fef2dd2-27af5a7b-127ba20c.tiff', 'REP': 'Opacities in the right upper or middle lobe, maybe early pneumonia.'},
    "FRO + REP -> LAT": {'FRO': 'e7f21453-7956d79a-44e44614-fae8ff16-d174d1a0.tiff', 'REP': 'No focal consolidation.', 'LAT': '8037e6b9-06367464-a4ccd63a-5c5c5a81-ce3e7ffc.tiff'},
    "LAT + REP -> FRO": {'LAT': '02c66644-b1883a91-54aed0e7-62d25460-398f9865.tiff', 'REP': 'No evidence of acute cardiopulmonary process.', 'FRO': 'b1f169f1-12177dd5-2fa1c4b1-7b816311-85d769e9.tiff'}
}


# CSS per personalizzare il tema
st.markdown("""
    <style>
    /* Sfondo scuro */
    body {
        background-color: #121212;
        color: white;
    }
    /* Personalizzazione del titolo */
    .title {
        font-size: 35px !important;
        font-weight: bold;
        color: #f63366;
    }
    /* Personalizzazione dei sottotitoli e testi principali */
    .stText, .stButton, .stMarkdown {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# Sostituisci questo con il link dell'immagine online
logo_1_path = "./DEMO/Loghi/Logo_UCBM.png"  # Sostituisci con il percorso del primo logo
logo_2_path = "./DEMO/Loghi/Logo UmU.png"  # Sostituisci con il percorso del secondo logo
logo_3_path = "./DEMO/Loghi/Logo COSBI.png"  # Sostituisci con il percorso del terzo logo
logo_4_path = "./DEMO/Loghi/logo trasparent.png"  # Sostituisci con il percorso del quarto logo
# Converti le immagini in base64
logo_1_base64 = image_to_base64(logo_1_path)
logo_2_base64 = image_to_base64(logo_2_path)
logo_3_base64 = image_to_base64(logo_3_path)
logo_4_base64 = image_to_base64(logo_4_path)

# CSS per posizionare i loghi in basso a destra e renderli piccoli
st.markdown(f"""
    <style>
    .footer {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 100;
        display: flex;
        gap: 10px; /* Spazio tra i loghi */
    }}
    .footer img {{
        height: 60px; /* Altezza dei loghi */
        width: auto; /* Mantiene il rapporto di aspetto originale */
    }}
    </style>
    <div class="footer">
        <img src="data:image/png;base64,{logo_1_base64}" alt="Logo 1">
        <img src="data:image/png;base64,{logo_2_base64}" alt="Logo 2">
        <img src="data:image/png;base64,{logo_3_base64}" alt="Logo 3">
        <img src="data:image/png;base64,{logo_4_base64}" alt="Logo 4">
    </div>
    """, unsafe_allow_html=True)

# Inizializzazione dello stato della sessione
if 'step' not in st.session_state:
    st.session_state['step'] = 1
if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None
if 'frontal_file' not in st.session_state:
    st.session_state['frontal_file'] = None
if 'lateral_file' not in st.session_state:
    st.session_state['lateral_file'] = None
if 'report' not in st.session_state:
    st.session_state['report'] = ""
if 'inputs' not in st.session_state:
    st.session_state['inputs'] = None
if 'outputs' not in st.session_state:
    st.session_state['outputs'] = None
if 'frontal' not in st.session_state:
    st.session_state['frontal'] = None
if 'lateral' not in st.session_state:
    st.session_state['lateral'] = None
if 'report' not in st.session_state:
    st.session_state['report'] = ""
if 'generate' not in st.session_state:
    st.session_state['generate'] = False

# Inizializza inference_tester solo una volta
if 'inference_tester' not in st.session_state:

    model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_video_diffuser_8frames.pth']
    st.session_state['inference_tester'] = dani_model(model='thesis_model',
                                                      data_dir='/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/checkpoints/',
                                                      pth=model_load_paths, load_weights=True)
    inference_tester = st.session_state['inference_tester']

    # Caricamento dei pesi Clip, Optimus, Frontal, Lateral e Text una sola volta
    if 'weights_loaded' not in st.session_state:
        clip_weights = 'Clip_Training/saved_checkpoints/Training_Clip_5e^-5/checkpoint_29_epoch_Training_Clip_5e^-5.pt'
        a, b = inference_tester.net.clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=False)

        optimus_weights = 'Report_Training/saved_checkpoints/VAE/checkpoint_99_epoch_VAE-Training-Prova1.pt'
        optimus_weights = torch.load(optimus_weights, map_location='cpu')
        a, b = inference_tester.net.optimus.load_state_dict(optimus_weights, strict=False)

        frontal_weights = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/CXR_Training/saved_checkpoints/Frontal/checkpoint_99_epoch_Training-Frontal-MultiPrompt-New.pt'
        frontal_weights = torch.load(frontal_weights, map_location='cpu')
        for key in list(frontal_weights.keys()):
            if 'unet_image' in key:
                value = frontal_weights.pop(key)
                new_key = key.replace('unet_image', 'unet_frontal')
                frontal_weights[new_key] = value

        lateral_weights = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/CXR_Training/saved_checkpoints/Lateral/checkpoint_99_epoch_Training-Lateral-MultiPrompt-New.pt'
        lateral_weights = torch.load(lateral_weights, map_location='cpu')
        for key in list(lateral_weights.keys()):
            if 'unet_image' in key:
                value = lateral_weights.pop(key)
                new_key = key.replace('unet_image', 'unet_lateral')
                lateral_weights[new_key] = value

        a, b = inference_tester.net.model.load_state_dict(frontal_weights, strict=False)
        a, b = inference_tester.net.model.load_state_dict(lateral_weights, strict=False)

        text_weights = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/Report_Training/saved_checkpoints/checkpoint_99_epoch_Report_Diffusion_Training-MultiPrompt-New.pt'
        text_weights = torch.load(text_weights, map_location='cpu')
        a, b = inference_tester.net.model.load_state_dict(text_weights, strict=False)
        st.session_state['weights_loaded'] = True  # Indica che i pesi sono stati caricati


# Usa inference_tester dalla sessione
inference_tester = st.session_state['inference_tester']

st.markdown('<h1 style="text-align: center" class="title">MedCoDi-M</h1>', unsafe_allow_html=True)

if st.session_state['step'] == 1:
    # Breve descrizione del lavoro
    st.markdown("""
        <div style='text-align: justify; font-size: 18px; line-height: 1.6;'>
            This thesis introduces MedCoDi-M, a novel multi-prompt vision-language model for multi-modal medical data generation. 
            In this demo, you will be able to perform various generation tasks including frontal and lateral chest X-rays and clinical report generation.
            MedCoDi-M enables flexible, any-to-any generation across different medical data modalities, utilizing contrastive learning and a modular approach for enhanced performance.
        </div>
    """, unsafe_allow_html=True)

    # lasciamo un po' di spazio
    st.markdown('<br>', unsafe_allow_html=True)

    image_path = "./DEMO/Loghi/model_final.png"  # Sostituisci con il percorso della tua immagine
    st.image(image_path, caption='Framework of MedCoDi-M', use_column_width=True)

    # Caption con dimensione del testo migliorata
    st.markdown("""
        <div style='text-align: center; font-size: 16px; font-style: italic; margin-top: 10px;'>
            Framework of MedCoDi-M: This demo allows you to generate frontal and lateral chest X-rays, as well as medical reports, through the MedCoDi-M model.
        </div>
    """, unsafe_allow_html=True)

    # lasciamo un po' di spazio
    st.markdown('<br>', unsafe_allow_html=True)

    # mettiamo un bottone con scritto Try it out
    if st.button("Try it out!"):
        st.session_state['step'] = 2
        st.rerun()

# Fase 1: Selezione dell'opzione
if st.session_state['step'] == 2:
    # Opzioni disponibili
    options = [
        "FRO -> LAT", "FRO -> REP", "FRO -> LAT + REP",
        "LAT -> FRO", "LAT -> REP", "LAT -> FRO + REP",
        "REP -> FRO", "REP -> LAT", "REP -> FRO + LAT",
        "FRO + LAT -> REP", "FRO + REP -> LAT", "LAT + REP -> FRO"
    ]

    # Messaggio di selezione con dimensione aumentata
    st.markdown(
        "<h4 style='text-align: justify'><strong>Select the type of generation you want to perform: (FRO = Frontal, LAT = Lateral, REP = Report)</strong></h4>",
        unsafe_allow_html=True)

    # Aumentare la dimensione di "Please select an option:"
    st.markdown(
        "<h4 style='text-align: justify'><strong>Please select an option:</strong></h4>",
        unsafe_allow_html=True)

    # Reset esplicito del valore di `selectbox` in caso di reset
    st.session_state['selected_option'] = st.selectbox(
        "", options, key='selectbox_option', index=0)  # Rimuoviamo il testo dal selectbox

    st.markdown('<br>', unsafe_allow_html=True)

    # Creiamo colonne per i pulsanti
    col1, col2, col3 = st.columns(3)

    # Pulsante per procedere con l'inferenza
    with col1:
        if st.button("Inference"):
            st.session_state['step'] = 3  # Passa al passo 3
            st.rerun()

    # Pulsante per provare un esempio
    with col2:
        if st.button("Try an example"):
            st.session_state['step'] = 5  # Passa al passo 5
            st.rerun()

    # Pulsante per tornare all'inizio
    with col3:
        if st.button("Return to the beginning"):
            # Ripristina lo stato della sessione
            st.session_state['step'] = 1
            st.session_state['selected_option'] = None
            st.session_state['selected_option2'] = None
            st.session_state['frontal_file'] = None
            st.session_state['lateral_file'] = None
            st.session_state['report'] = ""
            st.rerun()


# Fase 2: Caricamento file
if st.session_state['step'] == 3:
    st.markdown(
        f"<h4 style='text-align: justify'><strong>You selected: {st.session_state['selected_option']}. Now, please upload the required files below:</strong></h4>",
        unsafe_allow_html=True)

    # Carica l'immagine frontale
    if "FRO" in st.session_state['selected_option'].split(" ->")[0]:
        st.markdown("<h5 style='font-size: 18px;'>Load the Frontal X-ray in DICOM format</h5>", unsafe_allow_html=True)
        st.session_state['frontal_file'] = st.file_uploader("", type=["dcm"])

    # Carica l'immagine laterale
    if "LAT" in st.session_state['selected_option'].split(" ->")[0]:
        st.markdown("<h5 style='font-size: 18px;'>Load the Lateral X-ray in DICOM format</h5>", unsafe_allow_html=True)
        st.session_state['lateral_file'] = st.file_uploader("", type=["dcm"])

    # Inserisci il report clinico
    if "REP" in st.session_state['selected_option'].split(" ->")[0]:
        st.markdown("<h5 style='font-size: 18px;'>Type the clinical report</h5>", unsafe_allow_html=True)
        st.session_state['report'] = st.text_area("", value=st.session_state['report'])

        # lasciamo un po' di spazio
    st.markdown('<br>', unsafe_allow_html=True)

    # Creare colonne per allineare i pulsanti in orizzontale
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Generation"):
            frontal = None
            lateral = None
            report = None
            # Dato che questo step è velocissimo, prima di procedere mettiamo una finta barra di caricamento di 3 secondi
            with st.spinner("Preprocessing the data..."):
                time.sleep(3)
            # Controllo che i file necessari siano stati caricati
            if "FRO" in st.session_state['selected_option'].split(" ->")[0] and not st.session_state['frontal_file']:
                st.error("Load the Frontal image.")
            elif "LAT" in st.session_state['selected_option'].split(" ->")[0] and not st.session_state['lateral_file']:
                st.error("Load the Lateral image.")
            elif "REP" in st.session_state['selected_option'].split(" ->")[0] and not st.session_state['report']:
                st.error("Type the clinical report.")
            else:
                st.write(f"Execution of: {st.session_state['selected_option']}")

                # Carica l'immagine e avvia l'inferenza
                if st.session_state['frontal_file']:
                    dicom = pydicom.dcmread(st.session_state['frontal_file'])
                    image = dicom.pixel_array
                    if dicom.PhotometricInterpretation == 'MONOCHROME1':
                        image = (2 ** dicom.BitsStored - 1) - image
                    if dicom.ImagerPixelSpacing != [0.139, 0.139]:
                        zoom_factor = [0.139 / dicom.ImagerPixelSpacing[0], 0.139 / dicom.ImagerPixelSpacing[1]]
                        image = zoom(image, zoom_factor)
                    image = image / (2 ** dicom.BitsStored - 1)
                    # Se l'immagine non è quadrata, facciamo padding
                    if image.shape[0] != image.shape[1]:
                        diff = abs(image.shape[0] - image.shape[1])
                        pad_size = diff // 2
                        if image.shape[0] > image.shape[1]:
                            padded_image = np.pad(image, ((0, 0), (pad_size, pad_size)))
                        else:
                            padded_image = np.pad(image, ((pad_size, pad_size), (0, 0)))
                    # Resizing a 256x256 e a 512x512
                    zoom_factor = [256 / padded_image.shape[0], 256 / padded_image.shape[1]]
                    image_256 = zoom(padded_image, zoom_factor)
                    frontal = image_256
                    if frontal.dtype != np.uint8:
                        frontal2 = (255 * (frontal - frontal.min()) / (frontal.max() - frontal.min())).astype(np.uint8)
                    frontal = torch.tensor(frontal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    frontal2 = Image.fromarray(frontal2)
                    st.write("Frontal Image loaded successfully!")
                    # Mostra l'immagine caricata
                    st.image(frontal2, caption="Frontal Image Loaded", use_column_width=True)
                if st.session_state['lateral_file']:
                    dicom = pydicom.dcmread(st.session_state['lateral_file'])
                    image = dicom.pixel_array
                    if dicom.PhotometricInterpretation == 'MONOCHROME1':
                        image = (2 ** dicom.BitsStored - 1) - image
                    if dicom.ImagerPixelSpacing != [0.139, 0.139]:
                        zoom_factor = [0.139 / dicom.ImagerPixelSpacing[0], 0.139 / dicom.ImagerPixelSpacing[1]]
                        image = zoom(image, zoom_factor)
                    image = image / (2 ** dicom.BitsStored - 1)
                    # Se l'immagine non è quadrata, facciamo padding
                    if image.shape[0] != image.shape[1]:
                        diff = abs(image.shape[0] - image.shape[1])
                        pad_size = diff // 2
                        if image.shape[0] > image.shape[1]:
                            padded_image = np.pad(image, ((0, 0), (pad_size, pad_size)))
                        else:
                            padded_image = np.pad(image, ((pad_size, pad_size), (0, 0)))
                    # Resizing a 256x256 e a 512x512
                    zoom_factor = [256 / padded_image.shape[0], 256 / padded_image.shape[1]]
                    image_256 = zoom(padded_image, zoom_factor)
                    lateral = image_256
                    if lateral.dtype != np.uint8:
                        lateral2 = (255 * (lateral - lateral.min()) / (lateral.max() - lateral.min())).astype(np.uint8)
                    lateral = torch.tensor(lateral, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    lateral2 = Image.fromarray(lateral2)
                    st.write("Lateral Image loaded successfully!")
                    st.image(lateral2, caption="Lateral Image Loaded", use_column_width=True)
                if st.session_state['report']:
                    report = st.session_state['report']
                    st.write(f"Loaded Report: {report}")

                inputs = []
                if "FRO" in st.session_state['selected_option'].split(" ->")[0]:
                    inputs.append('frontal')
                if "LAT" in st.session_state['selected_option'].split(" ->")[0]:
                    inputs.append('lateral')
                if "REP" in st.session_state['selected_option'].split(" ->")[0]:
                    inputs.append('text')

                # Ora vediamo cosa c'è dopo la freccia
                outputs = []
                if "FRO" in st.session_state['selected_option'].split(" ->")[1]:
                    outputs.append('frontal')
                if "LAT" in st.session_state['selected_option'].split(" ->")[1]:
                    outputs.append('lateral')
                if "REP" in st.session_state['selected_option'].split(" ->")[1]:
                    outputs.append('text')

                # Ultima cosa che va fatta è passare allo step 4, prima di farlo però, tutte le variabili che ci servono
                # devono essere salvate nello stato della sessione
                st.session_state['inputs'] = inputs
                st.session_state['outputs'] = outputs
                st.session_state['frontal'] = frontal
                st.session_state['lateral'] = lateral
                st.session_state['report'] = report
                st.session_state['generate'] = True

                st.session_state['step'] = 4
                st.rerun()

    with col2:
        if st.button("Return to the beginning"):
            # Ripristina lo stato della sessione
            st.session_state['step'] = 1
            st.session_state['selected_option'] = None
            st.session_state['selected_option2'] = None
            st.session_state['frontal_file'] = None
            st.session_state['lateral_file'] = None
            st.session_state['report'] = ""
            st.rerun()

if st.session_state['step'] == 4:
    # Costruzione del prompt
    if st.session_state['generate'] is True:
        conditioning = []
        for inp in st.session_state['inputs']:
            if inp == 'frontal':
                cim = inference_tester.net.clip_encode_vision(st.session_state['frontal'], encode_type='encode_vision').to(device)
                uim = inference_tester.net.clip_encode_vision(torch.zeros_like(st.session_state['frontal']).to(device),
                                                              encode_type='encode_vision').to(device)
                conditioning.append(torch.cat([uim, cim]))
            elif inp == 'lateral':
                cim = inference_tester.net.clip_encode_vision(st.session_state['lateral'], encode_type='encode_vision').to(device)
                uim = inference_tester.net.clip_encode_vision(torch.zeros_like(st.session_state['lateral']).to(device),
                                                              encode_type='encode_vision').to(device)
                conditioning.append(torch.cat([uim, cim]))
            elif inp == 'text':
                ctx = inference_tester.net.clip_encode_text(1 * [st.session_state['report']], encode_type='encode_text').to(device)
                utx = inference_tester.net.clip_encode_text(1 * [""], encode_type='encode_text').to(device)
                conditioning.append(torch.cat([utx, ctx]))

        # Costruzione delle shapes
        shapes = []
        for out in st.session_state['outputs']:
            if out == 'frontal' or out == 'lateral':
                shape = [1, 4, 256 // 8, 256 // 8]
                shapes.append(shape)
            elif out == 'text':
                shape = [1, 768]
                shapes.append(shape)

        progress_bar = st.progress(0)

        # Inferenza
        z, _ = inference_tester.sampler.sample(
            steps=50,
            shape=shapes,
            condition=conditioning,
            unconditional_guidance_scale=7.5,
            xtype=st.session_state['outputs'],
            condition_types=st.session_state['inputs'],
            eta=1,
            verbose=False,
            mix_weight={'lateral': 1, 'text': 1, 'frontal': 1},
            progress_bar=progress_bar)

        # Decoder e visualizzazione dei risultati
        output_cols = st.columns(len(st.session_state['outputs']))

        # Definire due colonne per le immagini
        col1, col2 = st.columns(2)

        # Iterare sugli output e assegnare le immagini alle colonne corrispondenti
        for i, out in enumerate(st.session_state['outputs']):
            if out == 'frontal':
                x = inference_tester.net.autokl_decode(z[i])
                x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
                im = x[0].cpu().numpy()
                with col1:  # Mostrare la frontal image nella prima colonna
                    st.image(im, caption="Generated Frontal Image")
            elif out == 'lateral':
                x = inference_tester.net.autokl_decode(z[i])
                x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
                im = x[0].cpu().numpy()
                with col2:  # Mostrare la lateral image nella seconda colonna
                    st.image(im, caption="Generated Lateral Image")
            elif out == 'text':
                x = inference_tester.net.optimus_decode(z[i], max_length=100)
                x = [a.tolist() for a in x]
                rec_text = [inference_tester.net.optimus.tokenizer_decoder.decode(a) for a in x]
                rec_text = rec_text[0].replace('<BOS>', '').replace('<EOS>', '')
                st.write(f"Generated Report: {rec_text}")

        st.write("Generation completed successfully!")
        st.session_state['generate'] = False

    if st.button("Return to the beginning"):
        # Ripristina lo stato della sessione
        st.session_state['generate'] = False
        st.session_state['step'] = 1
        st.session_state['selected_option'] = None
        st.session_state['frontal_file'] = None
        st.session_state['lateral_file'] = None
        st.session_state['report'] = ""
        st.session_state['inputs'] = None
        st.session_state['outputs'] = None
        st.session_state['frontal'] = None
        st.session_state['lateral'] = None
        st.session_state['report'] = ""
        st.rerun()

if st.session_state['step'] == 5:
    st.markdown(
        f"<h4 style='text-align: justify'><strong>You selected: {st.session_state['selected_option']}</strong></h4>",
        unsafe_allow_html=True)

    inputs = []
    if "FRO" in st.session_state['selected_option'].split(" ->")[0]:
        inputs.append('FRO')
    if "LAT" in st.session_state['selected_option'].split(" ->")[0]:
        inputs.append('LAT')
    if "REP" in st.session_state['selected_option'].split(" ->")[0]:
        inputs.append('REP')

    outputs = []
    if "FRO" in st.session_state['selected_option'].split(" ->")[1]:
        outputs.append('FRO')
    if "LAT" in st.session_state['selected_option'].split(" ->")[1]:
        outputs.append('LAT')
    if "REP" in st.session_state['selected_option'].split(" ->")[1]:
        outputs.append('REP')

    esempio = esempi[st.session_state['selected_option']]

    # Mostra i file associati all'esempio
    st.markdown(
        "<h3 style='text-align: justify'><strong>INPUTS</strong></h3>",
        unsafe_allow_html=True)

    # Colonne per gli INPUTS
    input_cols = st.columns(len(inputs))

    for idx, inp in enumerate(inputs):
        with input_cols[idx]:
            if inp == 'FRO':
                path = "./DEMO/ESEMPI/" + esempio['FRO']
                print(path)
                if path.endswith(".tiff"):
                    im = tifffile.imread(path)
                    im = np.clip(im, 0, 1)
                elif path.endswith(".png"):
                    im = Image.open(path)
                st.image(im, caption="Frontal Image")
            elif inp == 'LAT':
                path = "./DEMO/ESEMPI/" + esempio['LAT']
                if path.endswith(".tiff"):
                    im = tifffile.imread(path)
                    im = np.clip(im, 0, 1)
                elif path.endswith(".png"):
                    im = Image.open(path)
                st.image(im, caption="Lateral Image")
            elif inp == 'REP':
                st.write(f"Report: {esempio['REP']}")

    st.markdown(
        "<h3 style='text-align: justify'><strong>OUTPUTS</strong></h3>",
        unsafe_allow_html=True)

    # Colonne per gli OUTPUTS
    output_cols = st.columns(len(outputs))

    for idx, out in enumerate(outputs):
        with output_cols[idx]:
            if out == 'FRO':
                path = "./DEMO/ESEMPI/" + esempio['FRO']
                if path.endswith(".tiff"):
                    im = tifffile.imread(path)
                    # facciamo clamp tra 0 e 1
                    im = np.clip(im, 0, 1)
                elif path.endswith(".png"):
                    im = Image.open(path)
                st.image(im, caption="Frontal Image")
            elif out == 'LAT':
                path = "./DEMO/ESEMPI/" + esempio['LAT']
                if path.endswith(".tiff"):
                    im = tifffile.imread(path)
                    # facciamo clamp tra 0 e 1
                    im = np.clip(im, 0, 1)
                elif path.endswith(".png"):
                    im = Image.open(path)
                st.image(im, caption="Lateral Image")
            elif out == 'REP':
                st.write(f"Report: {esempio['REP']}")

    # Pulsante per tornare all'inizio
    if st.button("Return to the beginning"):
        # Ripristina lo stato della sessione
        st.session_state['step'] = 1
        st.session_state['selected_option'] = None
        st.session_state['selected_option2'] = None
        st.session_state['frontal_file'] = None
        st.session_state['lateral_file'] = None
        st.session_state['report'] = ""
        st.rerun()

