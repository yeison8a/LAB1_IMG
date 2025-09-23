import cv2
import numpy as np
import matplotlib.pyplot as plt



pausar_video = False
velocidad = 33  # delay inicial en ms (~30 fps)

# Ruta del video
video_path = '../tiro1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: no se pudo abrir video")
    exit()
else:
    print("Video abierto correctamente")

# Tamaño de salida
nuevo_alto = 480
nuevo_ancho = 680

D_real_cm = 4.2   # diámetro real de la bola en cm
cm_por_px = None  # se calculará en el primer frame
trayectoria = []  # [(t, x_cm, y_cm)]
trayectoria_px = []
frames_id = []
x_val = []
y_val = []
vx_val = []
vy_val = []
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:  # fallback si el video no da fps
    fps = 30
tiempo = 0

while True:
    if not pausar_video:
        ret, frame = cap.read()
        if not ret:
            break

    # Redimensionar
    frame = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango para el rojo
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Máscara
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Suavizado + operaciones morfológicas
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Contornos
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) #Cual es la posicion del frame analizado
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 300:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if cm_por_px is None:
                    (x, y, w, h) = cv2.boundingRect(c)
                    diametro_px = (w + h) / 2
                    cm_por_px = D_real_cm / diametro_px

                x_cm = cx * cm_por_px
                y_cm = (nuevo_alto-cy) * cm_por_px
                (x, y, w, h) = cv2.boundingRect(c)
                radio = int((w + h) / 4)
                cv2.circle(frame, (cx, cy), radio, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                count = len(frames_id)

                if frames_id.count(f_id) ==0:
                    frames_id.append(f_id)
                    trayectoria.append((tiempo, x_cm, y_cm))
                    x_val.append(x_cm)
                    y_val.append(y_cm)
                    vx_val.append(0)
                    vy_val.append(0)
                    trayectoria_px.append(( cx, cy))
                    tiempo += 1 / fps
                else:
                    index = frames_id.index(f_id)
                    if len(frames_id)>1 and index>0:
                        vx_val[index] = (x_val[index] - x_val[index-1])*fps
                        vy_val[index] = (y_val[index] - y_val[index-1])*fps

    if frames_id.count(f_id) > 0:
        index = frames_id.index(f_id)
        cv2.putText(frame, f"X = {x_val[index]:0.2f}, Y= {y_val[index]:0.2f}   [cm]", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 100), 2)
        cv2.putText(frame, f"Vx = {vx_val[index]:0.2f}, Vy= {vy_val[index]:0.2f}   [cm/s]", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 100), 2)
    else:
        cv2.putText(frame, "X = ..., Y= ...", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 100), 2)
        cv2.putText(frame, "Vx = ..., Vy = ...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 100), 2)

    if len(frames_id)>1:
        for fr in range(len(frames_id[1:])):
            nfr = fr
            #print(trayectoria[fr][1:])
            cv2.line(frame, trayectoria_px[fr],trayectoria_px[fr+1],(30,60,60),2,cv2.LINE_AA)




    cv2.imshow("Video", frame)
    cv2.imshow("Mascara", mask)

    # Controles
    key = cv2.waitKey(velocidad) & 0xFF
    if key == 27:  # Esc
        break
    elif key == 32:  # Espacio pausa/reanuda
        pausar_video = not pausar_video
    elif key == ord('d'):  # Avanzar un frame
        pausar_video = True
        ret, frame = cap.read()
    elif key == ord('a'):  # Retroceder un frame
        pausar_video = True
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(pos-2, 0))
        ret, frame = cap.read()
    elif key == ord('+'):  # Aumentar velocidad
        velocidad = max(1, velocidad - 5)
        print(f"Velocidad aumentada: {1000/velocidad:.2f} fps")
    elif key == ord('-'):  # Disminuir velocidad
        velocidad += 5
        print(f"Velocidad reducida: {1000/velocidad:.2f} fps")

cap.release()
cv2.destroyAllWindows()

if len(trayectoria) > 2:
    trayectoria = np.array(trayectoria)
    t = trayectoria[:, 0]
    x = trayectoria[:, 1]
    y = trayectoria[:, 2]

    # Velocidades
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    v = np.sqrt(vx**2 + vy**2)

    # Aceleraciones
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    a = np.sqrt(ax**2 + ay**2)

    #aproximar la sln
    t_i= 2
    for accel in ay:
        if accel>0:
            break
        t_i+=1
    int_t = t[:t_i]
    calculate_x = vx[0] * int_t + x[0]
    calculate_y = -0.5 * 980 * int_t ** 2 + vy[0] * int_t + y[0]
    calculate_pos = np.sqrt(calculate_x**2 + calculate_y**2)

    vx_prom = np.mean(vx[:t_i-1])
    ay_prom = np.mean(ay[:t_i - 2])
    ax_prom = np.mean(ax[:t_i - 2])
    # Posición final
    print(f"accel prom {ax_prom:.2f}, {ay_prom:.2f}")
    #print("Velocidad promedio: %.2f cm/s" % vx_prom)
    #print("Aceleración promedio: %.2f cm/s^2" % np.mean(a))

    # Graficar trayectoria
    plt.figure()
    plt.plot(x, y, 'o-')
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title("Trayectoria")
    plt.grid()
    #plt.gca().invert_yaxis()

    # Graficar velocidad
    plt.figure(figsize=(6,4))
    plt.subplot(3,1,1)
    plt.plot(t[:t_i], v[:t_i], 'r-')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad (cm/s)")
    plt.title("Velocidad vs Tiempo")
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t[:t_i], vx[:t_i], 'r-')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad (cm/s)")
    plt.hlines(vx_prom,t[0],t[t_i],linestyles="dotted", label="Vx_prom")
    plt.title("Vx vs Tiempo")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t[:t_i], vy[:t_i])
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad (cm/s)")
    plt.title("Vy vs Tiempo")
    plt.grid()

    # Graficar aceleración
    plt.figure()
    plt.plot(t[:t_i-2], a[:t_i-2], 'g-', label="a")
    plt.plot(t[:t_i-2], ax[:t_i-2], 'r-.', label="ax")
    plt.plot(t[:t_i-2], ay[:t_i-2], 'b-.', label="ay")
    plt.hlines(ay_prom, t[0], t[t_i-3], linestyles="dotted", label="ay_prom")
    plt.hlines(ax_prom, t[0], t[t_i - 3], linestyles="dotted", label="ax_prom")
    plt.legend()
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Aceleración (cm/s^2)")
    plt.title("Aceleración vs Tiempo")
    plt.grid()

    #comparar resultados
    plt.figure()
    plt.plot(x[:t_i], y[:t_i], 'g-', label = "Pos real")
    plt.plot(calculate_x, calculate_y, 'r-.',label = "Pos aprox")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.legend()
    plt.title("Trayectoria aproximada")
    plt.grid()
    plt.show()

    x_final = x[t_i]
    y_final = y[t_i]
    x_final_1 = calculate_x[-1]
    y_final_1 = calculate_y[-1]
    if(len(x[:t_i]) != len(calculate_x)):
        print("NO Coinciden los vectores")

    print("Posición final: (%.2f cm, %.2f cm)" % (x_final, y_final))
    print("Posición final aprox: (%.2f cm, %.2f cm)" % (x_final_1, y_final_1))

else:
    print("No se detectó suficiente trayectoria para análisis.")
