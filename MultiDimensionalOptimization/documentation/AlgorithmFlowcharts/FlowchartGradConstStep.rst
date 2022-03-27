.. raw:: latex

    \begin{center}
        \begin{tikzpicture}[font=\small,thick]

          % Start block
          \node[draw,
            rounded rectangle,
            minimum width=2.5cm,
            minimum height=1cm] (block1) {start};

          % Voltage and Current Measurement
          \node[draw,
            trapezium,
            trapezium left angle = 65,
            trapezium right angle = 115,
            trapezium stretches,
            below=of block1,
            minimum width=3.5cm,
            minimum height=1cm
          ] (block2) { Given $\mathrm{x}_0$, $\varepsilon$, $\gamma$, $k_{\max}$};

          \node[draw,
            below=of block2,
            minimum width=3.5cm,
            minimum height=1cm
          ] (block3) {k = 0};

          % Power and voltage variation
          \node[draw,
            below=of block3,
            minimum width=3.5cm,
            minimum height=1cm
          ] (block4) {Evaluate $\nabla f(\mathrm{x}_k)$};

          % Conditions test
          \node[draw,
            diamond,
            below=1cm of block4,
            minimum width=3.5cm,
            inner sep=0] (block5) { \shortstack{$ \| \nabla f(\mathrm{x}_{k}) \|_2 < \varepsilon$ \\ $\mathbf{or} \ k \geq k_{\max} \quad$ }};


          \node[draw,
            right=3cm of block5,
            minimum width=4cm,
            minimum height=1cm,
            inner sep=0] (block6) { $\mathrm{x}_{k+1} = \mathrm{x}_{k} - \gamma \cdot \nabla f(\mathrm{x}_k)$};


          \node[draw,
            right=2.5cm of block4,
            minimum width=2cm,
            minimum height=1cm,
            inner sep=0] (block7) { $k = k + 1$};

          % Return block
          \node[draw,
            rounded rectangle,
            below=2cm of block5,
            minimum width=2.5cm,
            minimum height=1cm,] (block8) { return $\mathrm{x}_k$};


          % Arrows
          \draw[-latex] (block1) edge (block2)
          (block2) edge (block3)
          (block3) edge (block4)
          (block4) edge (block5)
          (block6) |- (block7)
          (block7) edge (block4);

          \draw[-stealth] (block5) -- (block6)
          node[pos=0.5,fill=white,inner sep=5]{No};

          \draw[-stealth] (block5) -- (block8)   node[pos=0.5,fill=white,inner sep=5]{Yes};
        \end{tikzpicture}

    \end{center}

