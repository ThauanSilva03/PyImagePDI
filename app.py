import io

import numpy as np
import streamlit as st
from PIL import Image

from core.algebric import cross_dissolve_uniform, cross_dissolve_non_uniform
from core.intensity import (
    negative_transformation,
    threshold_transformation,
    contrast_stretching,
    gamma_transformation,
    log_transformation,
)
from core.histogram import compute_histogram, expand_histogram, equalize_histogram
from core.spatial import mean_filter, median_filter
from core.convolution import (
    convolve2d,
    get_sharpening_kernel,
    get_emboss_kernels,
    get_mandatory_edge_kernels,
)
from core.edges import roberts_cross, sobel_operator, kirsch_operator, laplacian_edge
from core.sharpening import laplacian_sharpening, high_boost_filtering
from core.adaptive import adaptive_contrast_enhancement

st.set_page_config(
    page_title="PyImagePDI",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1300px; }

    /* Cabeçalho */
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        padding: 1.6rem 2rem;
        border-radius: 18px;
        color: #fff;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, .25);
    }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; color: #fff; }
    .hero p  { margin: .25rem 0 0; opacity: .9; font-size: 1rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
    section[data-testid="stSidebar"] .stSelectbox label { font-weight: 600; }

    /* Cards (containers com borda) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        border-color: rgba(124, 58, 237, .25);
        box-shadow: 0 2px 10px rgba(0, 0, 0, .05);
    }

    /* Botão principal */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        border: none;
        border-radius: 12px;
        padding: .7rem 1.2rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124, 58, 237, .35);
    }

    /* Imagens */
    div[data-testid="stImage"] img { border-radius: 10px; }
    div[data-testid="stFileUploader"] section { border-radius: 12px; border: 2px dashed #7c3aed66; }

    h2, h3 { font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🖼️ PyImagePDI</h1>
        <p>Imagens de 8 bits &middot; Escolha o grupo de operações, ajuste os parâmetros e clique em <b>Processar</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

ICONS = {
    "Operações Algébricas": "➕",
    "Transformações de Intensidade": "🎚️",
    "Histograma": "📊",
    "Filtros Espaciais": "🌫️",
    "Convolução Genérica e Testes": "🧮",
    "Detecção de Bordas": "✏️",
    "Aguçamento e High Boost": "🔍",
    "Contraste Adaptativo": "🌗",
}

st.sidebar.markdown("## 🧰 Menu")
operation_category = st.sidebar.selectbox(
    "Grupo de Operações",
    list(ICONS.keys()),
    format_func=lambda c: f"{ICONS[c]}  {c}",
)
st.sidebar.markdown("---")
st.sidebar.caption("Fluxo: **upload → parâmetros → Processar**")

TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

def load_gray(file):
    if hasattr(file, "seek"):
        file.seek(0)
    img = Image.open(file)
    if img.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        arr = np.array(img, dtype=np.float64)
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val > 255.0 or min_val < 0.0:
            if max_val > min_val:
                arr = (arr - min_val) / (max_val - min_val) * 255.0
            else:
                arr = np.zeros_like(arr)
        elif max_val <= 1.0 and max_val > 0.0 and img.mode == "F":
            arr = arr * 255.0
        return arr
    return np.array(img.convert("L"), dtype=np.float64)


def to_u8(arr):
    return np.clip(np.asarray(arr), 0, 255).astype(np.uint8)


def file_sig(*files):
    return tuple((f.name, f.size) for f in files)


def processar(key, fn, sig, label="Processar imagem"):
    st.write("")
    clicked = st.button(
        f"▶  {label}", type="primary", key=f"btn_{key}", use_container_width=True
    )
    if clicked:
        with st.spinner("⏳ Processando, aguarde..."):
            st.session_state[key] = {"sig": sig, "out": fn()}
        st.toast("Processamento concluído!", icon="✅")

    saved = st.session_state.get(key)
    if saved is None:
        st.info("Ajuste os parâmetros acima e clique em **Processar** para ver o resultado.")
        return None
    if saved["sig"] != sig:
        st.warning("Imagem ou parâmetros mudaram desde o último processamento. "
                   "Clique em **Processar** para atualizar.")
    return saved["out"]


def show(col, img, caption):
    col.image(img, caption=caption, use_container_width=True)


def download_png(arr, key, label="⬇️ Baixar resultado (PNG)"):
    buf = io.BytesIO()
    Image.fromarray(to_u8(arr)).save(buf, format="PNG")
    st.download_button(label, buf.getvalue(), file_name=f"{key}.png",
                       mime="image/png", key=f"dl_{key}")


def original_preview(img, caption="Pré-visualização (original)"):
    with st.container(border=True):
        st.image(to_u8(img), caption=caption, width=380)


if operation_category == "Operações Algébricas":
    st.header("➕ Dissolve Cruzado (Uniforme e Não Uniforme)")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            file1 = st.file_uploader("Imagem 1 (f1)", type=TYPES, key="file1")
        with col2:
            file2 = st.file_uploader("Imagem 2 (f2)", type=TYPES, key="file2")

    if file1 and file2:
        img1 = load_gray(file1)
        img2 = load_gray(file2)

        if img1.shape != img2.shape:
            st.warning("Imagens com resoluções diferentes. Redimensionando a Imagem 2 para coincidir com a Imagem 1.")
            img2 = np.array(
                Image.fromarray(img2.astype(np.uint8)).resize((img1.shape[1], img1.shape[0])),
                dtype=np.float64,
            )

        with st.container(border=True):
            dissolve_type = st.radio("Tipo de Dissolve", ["Uniforme", "Não Uniforme (Degradê)"], horizontal=True)
            if dissolve_type == "Uniforme":
                alpha = st.slider("Fator Alpha (Peso da Imagem 1)", 0.0, 1.0, 0.5, step=0.05)
                params = ("u", alpha)
                fn = lambda: cross_dissolve_uniform(img1, img2, alpha)
            else:
                direction = st.selectbox("Direção do Gradiente", ["horizontal", "vertical"])
                params = ("nu", direction)
                fn = lambda: cross_dissolve_non_uniform(img1, img2, direction)

        res = processar("alg", fn, (file_sig(file1, file2), params))

        c1, c2, c3 = st.columns(3)
        show(c1, to_u8(img1), "Imagem 1 (f1)")
        show(c2, to_u8(img2), "Imagem 2 (f2)")
        if res is not None:
            show(c3, res, "Resultado do Dissolve")
            download_png(res, "dissolve")
    else:
        st.info("📂 Carregue as duas imagens para continuar.")

elif operation_category == "Transformações de Intensidade":
    st.header("🎚️ Transformações Pontuais de Intensidade")

    with st.container(border=True):
        uploaded_file = st.file_uploader("Carregue uma Imagem", type=TYPES, key="intensity_file")

    if uploaded_file:
        img = load_gray(uploaded_file)

        with st.container(border=True):
            transform_type = st.selectbox(
                "Operação",
                ["Negativo", "Limiarização", "Alargamento de Contraste",
                 "Transformação Gama", "Transformação Logarítmica"],
            )

            if transform_type == "Negativo":
                st.caption("Fórmula: $s = 255 - r$")
                params, fn = (), lambda: negative_transformation(img)

            elif transform_type == "Limiarização":
                t_val = st.slider("Limiar (Threshold)", 0, 255, 128)
                st.caption(f"Fórmula: $s = 255$ se $r \\ge {t_val}$, senão $0$")
                params, fn = (t_val,), lambda: threshold_transformation(img, t_val)

            elif transform_type == "Alargamento de Contraste":
                st.caption(f"Faixa original: [{int(np.min(img))}, {int(np.max(img))}] → Mapeada para [0, 255]")
                params, fn = (), lambda: contrast_stretching(img)

            elif transform_type == "Transformação Gama":
                cg1, cg2 = st.columns(2)
                with cg1:
                    gamma = st.slider("Valor de Gamma (γ)", 0.05, 5.0, 1.0, step=0.05)
                with cg2:
                    c_const = st.slider("Constante c", 0.1, 2.0, 1.0, step=0.1)
                st.caption("Fórmula: $s = 255 \\cdot c \\cdot (r / 255)^\\gamma$")
                params, fn = (gamma, c_const), lambda: gamma_transformation(img, gamma, c_const)

            else:
                st.caption("Fórmula: $s = c \\cdot \\log(1 + r)$")
                params, fn = (), lambda: log_transformation(img)

        res = processar("int", fn, (file_sig(uploaded_file), transform_type, params))

        col_orig, col_res = st.columns(2)
        show(col_orig, to_u8(img), "Imagem Original")
        if res is not None:
            show(col_res, res, f"Resultado: {transform_type}")
            download_png(res, "intensidade")
    else:
        st.info("📂 Carregue uma imagem para continuar.")

elif operation_category == "Histograma":
    st.header("📊 Operações sobre Histograma (Expansão e Equalização)")

    with st.container(border=True):
        uploaded_file = st.file_uploader("Carregue uma Imagem", type=TYPES, key="hist_file")

    if uploaded_file:
        img = load_gray(uploaded_file)
        hist_orig = compute_histogram(img)

        with st.container(border=True):
            hist_op = st.radio(
                "Operação",
                ["Apenas Visualizar Histograma", "Expansão de Histograma", "Equalização de Histograma"],
                horizontal=True,
            )

        def run_hist():
            if hist_op == "Apenas Visualizar Histograma":
                return to_u8(img), hist_orig
            if hist_op == "Expansão de Histograma":
                r = expand_histogram(img)
            else:
                r = equalize_histogram(img)
            return r, compute_histogram(r)

        out = processar("hist", run_hist, (file_sig(uploaded_file), hist_op))

        col_i1, col_i2 = st.columns(2)
        show(col_i1, to_u8(img), "Imagem Original")
        if out is not None:
            res, hist_res = out
            show(col_i2, res, f"Resultado: {hist_op}")

            col_h1, col_h2 = st.columns(2)
            with col_h1:
                with st.container(border=True):
                    st.caption("Histograma Original")
                    st.bar_chart(hist_orig)
            with col_h2:
                with st.container(border=True):
                    st.caption("Histograma Processado")
                    st.bar_chart(hist_res)
            download_png(res, "histograma")
        else:
            with st.container(border=True):
                st.caption("Histograma Original")
                st.bar_chart(hist_orig)
    else:
        st.info("📂 Carregue uma imagem para continuar.")

elif operation_category == "Filtros Espaciais":
    st.header("🌫️ Filtragem Espacial (Média e Mediana)")

    with st.container(border=True):
        uploaded_file = st.file_uploader("Carregue uma Imagem", type=TYPES, key="spatial_file")

    if uploaded_file:
        img = load_gray(uploaded_file)

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                filter_type = st.selectbox("Tipo de Filtro", ["Filtro da Média (Linear)", "Filtro da Mediana (Não Linear)"])
            with c2:
                kernel_size = st.select_slider("Vizinhança (n x n)", options=[3, 5, 7, 9], value=3)
            with c3:
                border_mode = st.selectbox("Tratamento de Borda", ["reflect", "constants"])

        if "Média" in filter_type:
            fn = lambda: mean_filter(img, kernel_size, border_mode)
        else:
            fn = lambda: median_filter(img, kernel_size, border_mode)

        res = processar("spatial", fn, (file_sig(uploaded_file), filter_type, kernel_size, border_mode))

        col_orig, col_res = st.columns(2)
        show(col_orig, to_u8(img), "Imagem Original")
        if res is not None:
            show(col_res, res, f"{filter_type} ({kernel_size}x{kernel_size}, borda '{border_mode}')")
            download_png(res, "filtro_espacial")
    else:
        st.info("📂 Carregue uma imagem para continuar.")

elif operation_category == "Convolução Genérica e Testes":
    st.header("🧮 Convolução Genérica ($g = f * h + \\text{offset}$) e Testes Obrigatórios")

    with st.container(border=True):
        uploaded_file = st.file_uploader("Carregue uma Imagem", type=TYPES, key="conv_file")

    if uploaded_file:
        img = load_gray(uploaded_file)

        with st.container(border=True):
            cm1, cm2 = st.columns([2, 1])
            with cm1:
                test_mode = st.selectbox(
                    "Modo de Teste",
                    [
                        "Filtro de Aguçamento (com c e d)",
                        "Filtros de Relevo (h1 e h2)",
                        "Máscaras de Borda Obrigatórias (h3 e h4)",
                        "Máscara Customizada (Entrada Manual)",
                    ],
                )
            with cm2:
                border_mode = st.selectbox("Tratamento de Borda", ["reflect", "constants"], key="conv_border")

            if test_mode == "Filtro de Aguçamento (com c e d)":
                col_c, col_d, col_off = st.columns(3)
                with col_c:
                    c_val = st.number_input("Parâmetro c (inteiro positivo)", min_value=1, value=1, step=1)
                with col_d:
                    d_val = st.number_input("Parâmetro d (inteiro positivo)", min_value=1, value=1, step=1)
                with col_off:
                    offset_val = st.number_input("Offset", value=0.0, step=10.0)
                kernel = get_sharpening_kernel(c_val, d_val)
                st.write("**Máscara Aplicada:**")
                st.dataframe(kernel)
                st.caption(f"Soma dos coeficientes = {np.sum(kernel)} (deve ser igual a d={d_val})")

            elif test_mode == "Filtros de Relevo (h1 e h2)":
                h1, h2 = get_emboss_kernels()
                sel_h = st.radio("Máscara de Relevo", ["h1 (diagonal principal)", "h2 (diagonal secundária)"], horizontal=True)
                offset_val = st.slider("Offset (Recomendado: 128 para relevo cinza)", 0.0, 255.0, 128.0)
                kernel = h1 if sel_h.startswith("h1") else h2
                st.write("**Máscara Aplicada:**")
                st.dataframe(kernel)

            elif test_mode == "Máscaras de Borda Obrigatórias (h3 e h4)":
                h3, h4 = get_mandatory_edge_kernels()
                sel_b = st.radio("Máscara", ["h3 (gradiente direcional)", "h4 (Laplaciano alternativo)"], horizontal=True)
                offset_val = st.slider("Offset", 0.0, 255.0, 0.0)
                kernel = h3 if sel_b.startswith("h3") else h4
                st.write("**Máscara Aplicada:**")
                st.dataframe(kernel)

            else:
                st.write("Digite os elementos de uma máscara 3x3:")
                rows = []
                for i in range(3):
                    cols = st.columns(3)
                    rows.append([
                        cols[j].number_input(
                            f"k{i}{j}",
                            value=1.0 if (i == 1 and j == 1) else 0.0,
                            key=f"k{i}{j}",
                        )
                        for j in range(3)
                    ])
                kernel = np.array(rows, dtype=np.float64)
                offset_val = st.number_input("Offset Customizado", value=0.0)

        fn = lambda: convolve2d(img, kernel, offset=offset_val, border_mode=border_mode)
        sig = (file_sig(uploaded_file), test_mode, kernel.tobytes(), float(offset_val), border_mode)
        res = processar("conv", fn, sig)

        col_orig, col_res = st.columns(2)
        show(col_orig, to_u8(img), "Imagem Original")
        if res is not None:
            show(col_res, res, "Resultado da Convolução")
            download_png(res, "convolucao")
    else:
        st.info("📂 Carregue uma imagem para continuar.")

elif operation_category == "Detecção de Bordas":
    st.header("✏️ Detecção de Bordas (Roberts, Sobel, Kirsch e Laplaciano)")

    with st.container(border=True):
        uploaded_file = st.file_uploader("Carregue uma Imagem", type=TYPES, key="edge_file")

    if uploaded_file:
        img = load_gray(uploaded_file)

        with st.container(border=True):
            edge_method = st.selectbox(
                "Operador de Borda",
                ["Sobel (3x3)", "Roberts (2x2)", "Kirsch (8 Direções)", "Laplaciano"],
            )
            incl_diag = False
            if edge_method == "Laplaciano":
                incl_diag = st.checkbox("Incluir diagonais (Vizinhança 8, centro -8)", value=False)

        def run_edge():
            if edge_method == "Sobel (3x3)":
                return sobel_operator(img)
            if edge_method == "Roberts (2x2)":
                return roberts_cross(img)
            if edge_method == "Kirsch (8 Direções)":
                return kirsch_operator(img)
            return laplacian_edge(img, include_diagonals=incl_diag)

        out = processar("edge", run_edge, (file_sig(uploaded_file), edge_method, incl_diag))

        if out is None:
            original_preview(img)
        elif edge_method == "Sobel (3x3)":
            gx, gy, mag = out
            st.subheader("Respostas do Operador de Sobel")
            c1, c2, c3 = st.columns(3)
            show(c1, gx, "|Gx| (Derivada Horizontal / Bordas Verticais)")
            show(c2, gy, "|Gy| (Derivada Vertical / Bordas Horizontais)")
            show(c3, mag, "Magnitude Final: √(Gx² + Gy²)")
            download_png(mag, "sobel")

        elif edge_method == "Roberts (2x2)":
            r1, r2, mag = out
            st.subheader("Respostas do Operador de Roberts")
            c1, c2, c3 = st.columns(3)
            show(c1, r1, "|R1| (Diagonal Principal)")
            show(c2, r2, "|R2| (Diagonal Secundária)")
            show(c3, mag, "Magnitude Final: √(R1² + R2²)")
            download_png(mag, "roberts")

        elif edge_method == "Kirsch (8 Direções)":
            direcionais, mag_final = out
            st.subheader("Máscaras Direcionais de Kirsch (8 Orientações)")
            row1 = st.columns(4)
            for idx in range(4):
                nome, m_img = direcionais[idx]
                row1[idx].image(m_img, caption=f"Direção {nome}", use_container_width=True)
            row2 = st.columns(4)
            for idx in range(4, 8):
                nome, m_img = direcionais[idx]
                row2[idx - 4].image(m_img, caption=f"Direção {nome}", use_container_width=True)

            st.subheader("Resposta Final de Kirsch (Máximo entre as 8 direções)")
            st.image(mag_final, caption="Resultado Final: max(|f * K_k|)", width=450)
            download_png(mag_final, "kirsch")

        else:
            c_orig, c_lap = st.columns(2)
            show(c_orig, to_u8(img), "Imagem Original")
            show(c_lap, out, f"Bordas Laplaciano ({'Viz-8' if incl_diag else 'Viz-4'})")
            download_png(out, "laplaciano")
    else:
        st.info("📂 Carregue uma imagem para continuar.")

elif operation_category == "Aguçamento e High Boost":
    st.header("🔍 Aguçamento baseado no Laplaciano e Filtragem High Boost")

    with st.container(border=True):
        uploaded_file = st.file_uploader("Carregue uma Imagem", type=TYPES, key="sharp_file")

    if uploaded_file:
        img = load_gray(uploaded_file)

        with st.container(border=True):
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                method = st.selectbox("Método", ["Aguçamento Laplaciano Padrão (A = 1.0)", "High Boost (A >= 1.0)"])
            with col_m2:
                include_diag = st.checkbox("Incluir Diagonais no Laplaciano (Vizinhança-8, centro -8)", value=False)

            a_factor = 1.0
            if "Padrão" in method:
                st.caption("Fórmula: $g(x,y) = f(x,y) - \\nabla^2 f(x,y)$ (com centro da máscara negativo)")
                fn = lambda: laplacian_sharpening(img, include_diagonals=include_diag)
            else:
                a_factor = st.slider("Fator de Amplificação A", 1.0, 3.0, 1.3, step=0.1)
                st.caption(f"Fórmula: $g(x,y) = {a_factor} \\cdot f(x,y) - \\nabla^2 f(x,y)$")
                fn = lambda: high_boost_filtering(img, A=a_factor, include_diagonals=include_diag)

        out = processar("sharp", fn, (file_sig(uploaded_file), method, include_diag, a_factor))

        if out is None:
            original_preview(img)
        else:
            lap_vis, res = out
            st.subheader("Visualização dos Três Estágios")
            c1, c2, c3 = st.columns(3)
            show(c1, to_u8(img), "1. Imagem Original f(x,y)")
            show(c2, lap_vis, "2. Resposta do Laplaciano ∇²f (com offset 128)")
            show(c3, res, "3. Imagem Aguçada Combinada g(x,y)")
            download_png(res, "agucamento")
    else:
        st.info("📂 Carregue uma imagem para continuar.")

elif operation_category == "Contraste Adaptativo":
    st.header("🌗 Contraste Adaptativo (Média e Desvio Padrão Locais)")

    with st.container(border=True):
        uploaded_file = st.file_uploader("Carregue uma Imagem", type=TYPES, key="adaptive_file")

    if uploaded_file:
        img = load_gray(uploaded_file)

        with st.container(border=True):
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                kernel_size = st.select_slider("Vizinhança (n x n)", options=[3, 5, 7, 9, 11, 15], value=7)
            with col_c2:
                c_factor = st.slider("Constante de Ganho c", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
            with col_c3:
                border_mode = st.selectbox("Tratamento de Borda", ["reflect", "constants"], key="adapt_border")
            st.caption(
                f"Fórmula: $g = m + \\left(\\frac{{c \\cdot \\sigma_G}}{{\\sigma_L + \\epsilon}}\\right)(f - m)$ "
                f"com janela {kernel_size}x{kernel_size} e $c={c_factor}$."
            )

        fn = lambda: adaptive_contrast_enhancement(img, kernel_size=kernel_size, c=c_factor, border_mode=border_mode)
        res = processar("adaptive", fn, (file_sig(uploaded_file), kernel_size, c_factor, border_mode))

        col_orig, col_res = st.columns(2)
        show(col_orig, to_u8(img), "Imagem Original")
        if res is not None:
            show(col_res, res, f"Resultado Adaptativo (n={kernel_size}, c={c_factor})")
            download_png(res, "contraste_adaptativo")
    else:
        st.info("📂 Carregue uma imagem para continuar.")