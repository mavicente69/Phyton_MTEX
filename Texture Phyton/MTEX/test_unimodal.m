close all
%% 1. Configuración del Material (Zirconio - HCP)
% Definimos la simetría del cristal con los parámetros de red típicos del Zr
cs = crystalSymmetry('6/mmm', [3.232 3.232 5.147], 'mineral', 'Zirconium');

% Simetría de la muestra (usamos triclínica '1' para ver la mancha pura sin repeticiones)
ss = specimenSymmetry('1'); 

%% 2. Creación de la ODF Unimodal
% Orientación central en ángulos de Euler (Bunge ZXZ)
ori = orientation.byEuler(0*degree, 30*degree, 0*degree, cs, ss);
disp(quaternion(ori))

% Definimos el kernel campana estándar de MTEX con 15° de ancho
psi = deLaValleePoussinKernel('halfwidth', 7.5*degree);

% Generamos la ODF unimodal
odf_zr = unimodalODF(ori, psi);

%% 3. Gráfica de las Figuras de Polos
% Definimos los planos requeridos: Basal {0001}, Prismático {10-10} y Piramidal {10-11}
planos = Miller({0,0,0,1}, {1,0,-1,0}, {1,0,-1,1}, cs);

% Configuramos la ventana de dibujo y ploteamos
figure('Name', 'Figuras de Polos - Zr (0,30,0)', 'Color', 'w');
plotPDF(odf_zr, planos, 'contourf', 'Resolution', 2.5*degree);

% Ajustes estéticos (barra de colores y paleta)
mtexColorMap jet; 
mtexColorbar;