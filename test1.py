import tensorflow as tf

키 = 170
신발 = 260

a = tf.Variable(1.52)
b = tf.Variable(1.62)

def 손실함수():
  실제값 = 신발
  예측값 = 키 * a + b
  print(tf.square(실제값 - 예측값))
  return tf.square(실제값 - 예측값)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
  opt.minimize(손실함수, var_list=[a,b])
  print(a.numpy(),b.numpy())