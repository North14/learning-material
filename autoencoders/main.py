#!/usr/bin/env python3

from manim import *
import numpy as np

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init_layer(r, c):
    # random weights and biases between -1 and 1
    w = np.random.randn(r, c) * np.sqrt(2.0 / c)
    b = np.random.randn(r, 1) * 0.01
    return w, b

def mse_loss(p, t):
    return np.mean((p - t) ** 2)

def backpropagation(x, y, w1, b1, w2, b2, w3, b3, w4, b4, z1, a1, z2, a2, z3, a3, z4, a4, learning_rate=0.01):
    # Forward pass through 4 layers
    # z1 = np.dot(w1, x) + b1
    # a1 = relu(z1)
    # z2 = np.dot(w2, a1) + b2
    # a2 = relu(z2)
    # z3 = np.dot(w3, a2) + b3
    # a3 = relu(z3)
    # z4 = np.dot(w4, a3) + b4
    # a4 = z4

    # Compute loss (Mean Squared Error)
    loss = mse_loss(a4, y)

    # Backward pass
    dloss_da4 = 2 * (a4 - y) / y.size
    da4_dz4 = 1.0 # (z4).astype(float) # Linear derivative
    dloss_dz4 = dloss_da4 * da4_dz4

    dloss_dw4 = np.dot(dloss_dz4, a3.T)
    dloss_db4 = np.sum(dloss_dz4, axis=1, keepdims=True)

    dloss_da3 = np.dot(w4.T, dloss_dz4)
    da3_dz3 = leaky_relu_derivative(z3)
    # da3_dz3 = (z3 > 0).astype(float) # ReLU derivative
    # da3_dz3 = sigmoid(z3) * (1 - sigmoid(z3))
    # da3_dz3 = relu(z3)
    dloss_dz3 = dloss_da3 * da3_dz3

    dloss_dw3 = np.dot(dloss_dz3, a2.T)
    dloss_db3 = np.sum(dloss_dz3, axis=1, keepdims=True)

    dloss_da2 = np.dot(w3.T, dloss_dz3)
    #da2_dz2 = (z2 > 0).astype(float) # ReLU derivative
    # da2_dz2 = sigmoid(z2) * (1 - sigmoid(z2))
    da2_dz2 = leaky_relu_derivative(z2)
    dloss_dz2 = dloss_da2 * da2_dz2

    dloss_dw2 = np.dot(dloss_dz2, a1.T)
    dloss_db2 = np.sum(dloss_dz2, axis=1, keepdims=True)

    dloss_da1 = np.dot(w2.T, dloss_dz2)
    # da1_dz1 = (z1 > 0).astype(float) # ReLU derivative
    # da1_dz1 = sigmoid(z1) * (1 - sigmoid(z1))
    da1_dz1 = leaky_relu_derivative(z1)
    dloss_dz1 = dloss_da1 * da1_dz1

    dloss_dw1 = np.dot(dloss_dz1, x.T)
    dloss_db1 = np.sum(dloss_dz1, axis=1, keepdims=True)

    # Update weights and biases (Gradient Descent)
    w4 -= learning_rate * dloss_dw4
    b4 -= learning_rate * dloss_db4
    w3 -= learning_rate * dloss_dw3
    b3 -= learning_rate * dloss_db3
    w2 -= learning_rate * dloss_dw2
    b2 -= learning_rate * dloss_db2
    w1 -= learning_rate * dloss_dw1
    b1 -= learning_rate * dloss_db1

    return loss, w1, b1, w2, b2, w3, b3, w4, b4

def gaussian_dataset(num_samples, num_features):
    # Generate around 3 num_features clusters
    # randomly choose datapoints from gaussian distributions
    data = []
    num_clusters = 2
    cluster_centers = np.random.randn(num_clusters, num_features) * 2
    for _ in range(num_samples):
        cluster = np.random.randint(0, num_clusters)
        point = cluster_centers[cluster] + np.random.randn(num_features)
        data.append(point)
    return np.array(data).T  # Shape (num_features, num_samples)

class VisualizeAutoencoder(Scene):
    def create_nodes(self, left_shift, down_shift, num_nodes, layer_output=None):
        node_group = VGroup()
        nodes = []
        
        for i in range(num_nodes):
            opacity = 0.0
            text = "0.0"

            if layer_output is not None and np.max(layer_output) != 0.0:
                opacity = np.abs(float(layer_output[i][0])) / np.max(np.abs(layer_output))
                text = f"{layer_output[i][0]:.2f}"

            node = Circle(radius=0.35, stroke_color=WHITE, stroke_width=0.5, fill_color=GRAY, fill_opacity=opacity)

            label = Text(text, font_size=18).move_to(node.get_center())
            group = VGroup(node, label)
            
            nodes.append(group)
            node_group.add(group)

        # Arrange
        node_group.arrange(DOWN, buff=0.5)
        node_group.shift(LEFT * left_shift + DOWN * down_shift)

        return node_group, nodes

    def create_connections(self, from_nodes, to_nodes, weights):
        connections = VGroup()
        max_abs_weight = np.max(np.abs(weights)) + 1e-5
        for i, from_node in enumerate(from_nodes):
            for j, to_node in enumerate(to_nodes):
                weight = weights[j][i]
                # opacity = min(max((weight - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-5), 0), 1)
                opacity = np.abs(weight) / max_abs_weight
                colour = BLUE if weight >= 0 else RED
                line = Line(from_node.get_edge_center(RIGHT), to_node.get_edge_center(LEFT), stroke_color=colour, stroke_width=2, stroke_opacity=opacity)
                connections.add(line)
        return connections

    def construct(self):
        # Parameters
        input_size = 6
        hidden_size = 4
        choke_size = 3
        output_size = 6

        # Initialize weights and biases
        w1, b1 = init_layer(hidden_size, input_size)
        w2, b2 = init_layer(choke_size, hidden_size)
        w3, b3 = init_layer(hidden_size, choke_size)
        w4, b4 = init_layer(output_size, hidden_size)

        # Sample input data
        # input_data = np.random.randint(100, size=(6, 10)) / 100  # Shape (6, 3)
        # generate a random large gaussian dataset
        input_data = gaussian_dataset(num_samples=5000, num_features=input_size)  # Shape (6, 10)

        lshift = lambda x: 2 * x + 0.5


        in_node_group, in_nodes = self.create_nodes(left_shift=lshift(2), down_shift=0, num_nodes=input_size, layer_output=input_data[:,0:1])
        h1_node_group, h1_nodes = self.create_nodes(left_shift=lshift(1), down_shift=0, num_nodes=hidden_size)
        choke_node_group, choke_nodes = self.create_nodes(left_shift=lshift(0), down_shift=0, num_nodes=choke_size)
        h2_node_group, h2_nodes = self.create_nodes(left_shift=lshift(-1), down_shift=0, num_nodes=hidden_size)
        out_node_group, out_nodes = self.create_nodes(left_shift=lshift(-2), down_shift=0, num_nodes=output_size)


        self.play(Create(in_node_group),
                  Create(h1_node_group),
                  Create(choke_node_group),
                  Create(h2_node_group),
                  Create(out_node_group))
        self.wait(1)

        # Connections
        conn1 = self.create_connections(in_nodes, h1_nodes, w1)
        conn2 = self.create_connections(h1_nodes, choke_nodes, w2)
        conn3 = self.create_connections(choke_nodes, h2_nodes, w3)
        conn4 = self.create_connections(h2_nodes, out_nodes, w4)
        self.play(Create(conn1), Create(conn2), Create(conn3), Create(conn4))
        self.wait(1)

        # Information labels and screens
        phase_text = Text("Training Autoencoder", font_size=30).to_edge(UP)
        input_text = Text("Input", font_size=24).next_to(in_node_group, UP)
        hidden1_text = Text("Hidden Layer", font_size=24).next_to(h1_node_group, UP)
        choke_text = Text("Bottleneck", font_size=24).next_to(choke_node_group, UP)
        hidden2_text = Text("Hidden Layer", font_size=24).next_to(h2_node_group, UP)
        output_text = Text("Output", font_size=24).next_to(out_node_group, UP)

        info_box = Rectangle(width=2, height=1, stroke_color=WHITE, fill_color=BLACK, fill_opacity=0.5)
        info_box.to_corner(UR, buff=0.5)
        info_text = Text("Epoch: 0\nLoss: 0.00", font_size=18).move_to(info_box.get_center())

        self.play(
            Write(phase_text),
            Write(input_text),
            Write(hidden1_text),
            Write(choke_text),
            Write(hidden2_text),
            Write(output_text)
        )
        self.play(Create(info_box), Write(info_text))
        self.wait(1)

        epochs = 5 # 35
        loss_history = []
        prev_text_box = VGroup()
        # Training loop
        for epoch in range(epochs):
        # Forward propagation
            # take a sample of 50 datapoints from input
            sample_indices = np.random.choice(input_data.shape[1], 50, replace=False)
            input_sample = input_data[:, sample_indices]

            average_loss_epoch = 0
            for i in range(input_sample.shape[1]):
                x = input_sample[:, i:i+1]
                # Forward pass
                z1 = np.dot(w1, x) + b1
                a1 = leaky_relu(z1)
                z2 = np.dot(w2, a1) + b2
                a2 = leaky_relu(z2)
                z3 = np.dot(w3, a2) + b3
                a3 = leaky_relu(z3)
                z4 = np.dot(w4, a3) + b4
                a4 = z4 # Linear output

                # Backpropagation (update all layers)
                loss, w1, b1, w2, b2, w3, b3, w4, b4 = backpropagation(x, x, w1, b1, w2, b2, w3, b3, w4, b4,
                                                                        z1, a1, z2, a2, z3, a3, z4, a4,
                                                                        learning_rate=0.01)
                
                # update the color of connections based on new weights by recreating connection groups
                new_conn1 = self.create_connections(in_nodes, h1_nodes, w1)
                new_conn2 = self.create_connections(h1_nodes, choke_nodes, w2)
                new_conn3 = self.create_connections(choke_nodes, h2_nodes, w3)
                new_conn4 = self.create_connections(h2_nodes, out_nodes, w4)

                new_animations = []
                node_layers = [
                    (in_nodes, x),
                    (h1_nodes, a1),
                    (choke_nodes, a2),
                    (h2_nodes, a3),
                    (out_nodes, a4)
                        ]
                for node, activation in node_layers:
                    for idx, node in enumerate(node):
                        opacity = 0.0
                        text = "0.0"
                        if np.max(activation) != 0.0:
                            opacity = np.clip(np.abs(float(activation[idx][0]) / np.max(activation)), 0, 1)
                            text = f"{activation[idx][0]:.2f}"
                        new_label = Text(text, font_size=18).move_to(node.get_center())
                        new_animations.append(node[0].animate.set_fill(opacity=opacity))
                        new_animations.append(Transform(node[1], new_label))
                    
                # Update info box
                yloss = mse_loss(a4, x)
                # yloss = np.mean((a4 - x) ** 2)

                # Animate the changes
                if i % 15 == 0:
                    self.play(
                            *new_animations,
                            Transform(conn1, new_conn1),
                            Transform(conn2, new_conn2),
                            Transform(conn3, new_conn3),
                            Transform(conn4, new_conn4),
                            Transform(info_text, Text(f"Epoch: {epoch+1}\nLoss: {yloss:.2f}", font_size=18).move_to(info_box.get_center())),
                            run_time=0.2
                            )
                average_loss_epoch += yloss

            avg_loss = average_loss_epoch / input_sample.shape[1]
            loss_history.append(avg_loss)
            text_list = []

            # take the current ten last losses and get their epoch
            for e in range(max(0, epoch-9), epoch+1):
                loss_history_str = f"Epoch {e+1}: Loss = {loss_history[e]:.2f}\n"
                text_list.append(Text(loss_history_str, font_size=14))
            text_box = VGroup(*text_list).arrange(DOWN, aligned_edge=LEFT)
            # update the loss history box and remove the previous one
            text_box.next_to(info_box, DOWN)
            text_box.set_color_by_gradient(GRAY, WHITE)
            # Animate the creation or transformation
            if epoch == 0:
                self.play(Create(text_box), run_time=0.2)
            else:
                # Transform the previous box into the new, correctly positioned one
                self.play(ReplacementTransform(prev_text_box, text_box), run_time=0.4)
            # Store the current box for the next loop's Transform
            prev_text_box = text_box
        self.wait(2)

        # Testing on a mix of data between the old input and a sample of new and unseen data
        # Old data = benign samples, New data = anomalous samples
        # 1. Sample old data and generate new data, then mix them with a given ration
        # add a y label for later reconstruction error chart
        y_labels = []
        anomaly_ratio = 0.2
        num_test_samples = 100
        old_input = np.random.choice(input_data.shape[1], int(num_test_samples*(1 - anomaly_ratio)), replace=False)
        new_input = gaussian_dataset(num_samples=int(num_test_samples*anomaly_ratio), num_features=input_size)
        test_input = input_data[:, old_input]
        test_input = np.concatenate((test_input, new_input), axis=1)  # Shape (6, num_test_samples)
        # Shuffle the test input columns
        np.random.shuffle(test_input.T)
        # Create y_labels for the reconstruction error chart
        y_labels = [0] * int(num_test_samples * (1 - anomaly_ratio)) + [1] * int(num_test_samples * anomaly_ratio)
        np.random.shuffle(y_labels)

        # 2. Update phase_text to testing
        phase_text_new = Text("Testing Autoencoder", font_size=30).to_edge(UP)
        self.play(Transform(phase_text, phase_text_new))
        self.wait(3)
        
        # 3. Run the data through the autoencoder.
        reconstruction_errors = []
        for i in range(test_input.shape[1]):
            x = test_input[:, i:i+1]
            # Forward pass
            z1 = np.dot(w1, x) + b1
            a1 = leaky_relu(z1)
            z2 = np.dot(w2, a1) + b2
            a2 = leaky_relu(z2)
            z3 = np.dot(w3, a2) + b3
            a3 = leaky_relu(z3)
            z4 = np.dot(w4, a3) + b4
            a4 = z4 # Linear output

            reconstruction_errors.append(mse_loss(a4, x))

            # update the color of connections based on new weights by recreating connection groups
            new_conn1 = self.create_connections(in_nodes, h1_nodes, w1)
            new_conn2 = self.create_connections(h1_nodes, choke_nodes, w2)
            new_conn3 = self.create_connections(choke_nodes, h2_nodes, w3)
            new_conn4 = self.create_connections(h2_nodes, out_nodes, w4)

            new_animations = []
            node_layers = [
                (in_nodes, x),
                (h1_nodes, a1),
                (choke_nodes, a2),
                (h2_nodes, a3),
                (out_nodes, a4)
                    ]
            for node, activation in node_layers:
                for idx, node in enumerate(node):
                    opacity = 0.0
                    text = "0.0"
                    if np.max(activation) != 0.0:
                        opacity = np.clip(np.abs(float(activation[idx][0]) / np.max(activation)), 0, 1)
                        text = f"{activation[idx][0]:.2f}"
                    new_label = Text(text, font_size=18).move_to(node.get_center())
                    new_animations.append(node[0].animate.set_fill(opacity=opacity))
                    new_animations.append(Transform(node[1], new_label))
                
            # Animate the changes
            if i % 5 == 0:
                self.play(
                        *new_animations,
                        Transform(conn1, new_conn1),
                        Transform(conn2, new_conn2),
                        Transform(conn3, new_conn3),
                        Transform(conn4, new_conn4),
                        run_time=0.2
                        )
        
        # 4. Show a chart of reconstruction errors, coloring benign samples in green and anomalous samples in red.
        # Calculate reconstruction errors, the threshold should be 6*MAD
        reconstruction_errors = np.array(reconstruction_errors)
        median_error = np.median(reconstruction_errors)
        mad = np.median(np.abs(reconstruction_errors - median_error))
        threshold = median_error + mad
        # Replace neural network with Create the bar chart and confusion matrix 
        self.play(
            FadeOut(in_node_group),
            FadeOut(h1_node_group),
            FadeOut(choke_node_group),
            FadeOut(h2_node_group),
            FadeOut(out_node_group),
            FadeOut(conn1),
            FadeOut(conn2),
            FadeOut(conn3),
            FadeOut(conn4),
            FadeOut(input_text),
            FadeOut(hidden1_text),
            FadeOut(choke_text),
            FadeOut(hidden2_text),
            FadeOut(output_text),
            FadeOut(info_box),
            FadeOut(info_text),
            FadeOut(prev_text_box),
            FadeOut(phase_text_new),
            FadeOut(phase_text)
        )
        self.wait(1)
        # Create bar chart
        vals, colors = [], []
        for i, error in enumerate(reconstruction_errors):
            height = error # Scale for better visibility
            color = GREEN if error <= threshold else RED
            vals.append(height)
            colors.append(color)
        bar_group = BarChart(
            vals,
            bar_colors=colors,
            bar_names=None,
        )
        threshold_line = Line(
            start=bar_group.coords_to_point(-2, threshold),
            end=bar_group.coords_to_point(len(reconstruction_errors)+2, threshold),
            stroke_width=2,
            color=YELLOW
        )
        title = Text(f"Reconstruction Loss in MSE (Threshold = {threshold:.2f})", font_size=36).to_edge(UP)
        bottom_title = Text("Green: Benign Samples, Red: Anomalous Samples", t2c={ "Green": GREEN, "Red": RED }, font_size=24).to_edge(DOWN)
        subtitle = Text(f"Anomaly Threshold = Median + MAD", t2c={"Anomaly Threshold": YELLOW}, font_size=24).next_to(title, DOWN)
    
        self.play(Write(title),
                  Write(bottom_title),
                  Write(subtitle))
        self.play(Create(bar_group),
                  Create(threshold_line))

        # 5. End scene.

        self.wait(10)

class TestingBarChart(Scene):
    def construct(self):
        # Sample reconstruction errors and labels
        reconstruction_errors = np.random.rand(100) * 0.5
        y_labels = [0] * 80 + [1] * 20  # 80 benign, 20 anomalous
        np.random.shuffle(y_labels)

        # Calculate threshold using MAD
        median_error = np.median(reconstruction_errors)
        mad = np.median(np.abs(reconstruction_errors - median_error))
        threshold = median_error + mad

        # Create bar chart
        vals, colors, labels = [], [], []
        for i, error in enumerate(reconstruction_errors):
            height = error  # Scale for better visibility
            color = GREEN if error <= threshold else RED
            vals.append(height)
            colors.append(color)
            labels.append(str(i))

        bar_group = BarChart(
            vals,
            bar_colors=colors,
            bar_names=None,
        )
        # Get the corresponding coordinate for the given threshold value inside the bar chart
        # bar_group.coords_to_point(x, y)
        threshold_line = Line(
            start=bar_group.coords_to_point(-3, threshold),
            end=bar_group.coords_to_point(len(reconstruction_errors)+3, threshold),
            stroke_width=2,
            color=YELLOW
        )
        self.play(Create(threshold_line))
        # Write the current threshold value
        title = Text(f"Anomaly detection", font_size=36).to_edge(UP)
        subtitle = Text(f"Threshold (MAD): {threshold:.3f}", font_size=24).next_to(title, DOWN)
        bottom_title = Text("Green: Benign Samples, Red: Anomalous Samples", t2c={ "Green": GREEN, "Red": RED }, font_size=24).to_edge(DOWN)

        self.play(Write(title),
                  Write(subtitle),
                  Write(bottom_title))
        self.play(Create(bar_group))
        self.wait(5)

class ChangeBarValuesExample(Scene):
    def construct(self):
        values=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]

        chart = BarChart(
            values,
            y_range=[-10, 10, 2],
            y_axis_config={"font_size": 24},
        )
        self.add(chart)

        chart.change_bar_values(list(reversed(values)))
        self.add(chart.get_bar_labels(font_size=24))