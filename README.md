#Sprint IOT

#Integrantes
- Victor Camargo Maciel RM98384
- Geovanna Silva Cunha RM97736
- João Arthur Monteiro Pajaro RM551272
- Gustavo Medeiros RM552093

# 🔒 Reconhecimento Facial Local com OpenCV (LBPH)

Este projeto implementa um **sistema local de reconhecimento facial** usando **OpenCV Haar Cascade** para detecção e **LBPH** (Local Binary Patterns Histograms) para reconhecimento.  
O código funciona **offline**, sem necessidade de conexão com a nuvem, e pode opcionalmente enviar sinais para um **Arduino** via porta serial (ex.: abrir uma trava, acender um LED etc.).

---

## 📌 Funcionalidades
- Detecta rostos usando **Haar Cascade** (OpenCV).
- Cadastra novos usuários com múltiplas amostras.
- Reconhece rostos cadastrados e exibe apenas:
  - **Nome** → se cadastrado.
  - **Desconhecido** → se não estiver no banco.
- Envia um sinal `O` via **Serial (COM)** quando identifica alguém válido (cooldown para evitar repetições).
- Funciona totalmente **offline**.

---

## 🛠️ Dependências

Instale os pacotes necessários:

```bash
pip install opencv-contrib-python numpy pyserial


