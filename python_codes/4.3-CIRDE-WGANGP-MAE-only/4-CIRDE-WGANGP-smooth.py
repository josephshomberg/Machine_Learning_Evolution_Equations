# ============================================================
# MAE-Only Inverse Training (Chafee--Infante)
# ============================================================

# --- Data paths ---
TRAIN_NPZ = "YOUR/PATH/TO/TRAINING-DATASET.npz"
TEST_NPZ  = "YOUR/PATH/TO/TESTING-DATASET.npz"

# --- Loss weights ---
LAMBDA_MAE = 25.0
LAMBDA_ADV = 2.0
LAMBDA_GP  = 10.0

# --- Training parameters ---
BATCH_SIZE   = 8
MAX_EPOCHS   = 20
CRITIC_STEPS = 4

# ============================================================
# Load and normalize data
# ============================================================

train_npz = np.load(TRAIN_NPZ, mmap_mode='r')
Xsrc = train_npz['src']   # u_T
Xtar = train_npz['tar']   # u_0

NE_SCALE = np.max(np.abs(Xsrc))
IC_SCALE = np.max(np.abs(Xtar))

trainA = (Xsrc / NE_SCALE)[..., None]
trainB = (Xtar / IC_SCALE)[..., None]

dataset = [trainA, trainB]
image_shape = trainA.shape[1:]

# ============================================================
# Generator (U-Net style)
# ============================================================

def generator(image_shape):
    in_image = Input(shape=image_shape)

    e1 = Conv2D(64, 3, strides=2, padding='same')(in_image)
    e1 = LeakyReLU(0.2)(e1)

    e2 = Conv2D(128, 3, strides=2, padding='same')(e1)
    e2 = LayerNormalization()(e2)
    e2 = LeakyReLU(0.2)(e2)

    b = Conv2D(256, 3, strides=2, padding='same')(e2)
    b = Activation('relu')(b)

    d1 = Conv2DTranspose(128, 3, strides=2, padding='same')(b)
    d1 = Concatenate()([d1, e2])

    d2 = Conv2DTranspose(64, 3, strides=2, padding='same')(d1)
    d2 = Concatenate()([d2, e1])

    out = Conv2DTranspose(1, 3, strides=2, padding='same')(d2)
    out = Activation('tanh')(out)

    return Model(in_image, out)

# ============================================================
# Critic (PatchGAN)
# ============================================================

def critic(image_shape):
    in_src = Input(shape=image_shape)
    in_tgt = Input(shape=image_shape)

    x = Concatenate()([in_src, in_tgt])
    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    out = Conv2D(1, 4, padding='same')(x)
    return Model([in_src, in_tgt], out)

# ============================================================
# Gradient penalty (WGAN-GP)
# ============================================================

def gradient_penalty(c_model, real_A, real_B, fake_B):
    alpha = tf.random.uniform([tf.shape(real_B)[0], 1, 1, 1], 0., 1.)
    interp = alpha * real_B + (1.0 - alpha) * fake_B

    with tf.GradientTape() as tape:
        tape.watch(interp)
        pred = c_model([real_A, interp], training=True)
        pred = tf.reduce_sum(pred, axis=[1,2,3])

    grads = tape.gradient(pred, interp)
    grads = tf.reshape(grads, [tf.shape(grads)[0], -1])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    return tf.reduce_mean((slopes - 1.0)**2)

# ============================================================
# Training step
# ============================================================

@tf.function
def train_step(real_A, real_B, generator, critic, g_opt, c_opt):

    # --- Critic update ---
    for _ in tf.range(CRITIC_STEPS):
        with tf.GradientTape() as tape:
            fake_B = generator(real_A, training=True)

            c_real = critic([real_A, real_B], training=True)
            c_fake = critic([real_A, fake_B], training=True)

            gp = gradient_penalty(critic, real_A, real_B, fake_B)

            c_loss = tf.reduce_mean(c_fake) - tf.reduce_mean(c_real) + LAMBDA_GP * gp

        grads = tape.gradient(c_loss, critic.trainable_variables)
        c_opt.apply_gradients(zip(grads, critic.trainable_variables))

    # --- Generator update ---
    with tf.GradientTape() as tape:
        fake_B = generator(real_A, training=True)
        c_fake = critic([real_A, fake_B], training=True)

        adv_loss = -tf.reduce_mean(c_fake)
        mae_loss = tf.reduce_mean(tf.abs(real_B - fake_B))

        g_loss = LAMBDA_ADV * adv_loss + LAMBDA_MAE * mae_loss

    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(grads, generator.trainable_variables))

# ============================================================
# Train loop
# ============================================================

c_model = critic(image_shape)
g_model = generator(image_shape)

g_opt = Adam(1e-4, beta_1=0.0, beta_2=0.9)
c_opt = Adam(2e-4, beta_1=0.0, beta_2=0.9)

for epoch in range(MAX_EPOCHS):
    for _ in range(len(trainA) // BATCH_SIZE):
        idx = np.random.randint(0, trainA.shape[0], BATCH_SIZE)
        real_A = trainA[idx]
        real_B = trainB[idx]

        train_step(real_A, real_B, g_model, c_model, g_opt, c_opt)
