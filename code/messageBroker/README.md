# Dealing with the mosquitto message broker

## Manage user accounts

### Create users

#### Method 1 - manually dealing with simple text file

1. Create text file `~/repositories/AI-CPS/code/messageBroker/users.txt`.
It shall have the following format:

```
   user1:password1
   user2:password2
```

2. Encrypt passwords of this text file:

```
   mosquitto_passwd -U ~/repositories/AI-CPS/code/messageBroker/users.txt
```

3. The password file is now ready to use.
So, for instance start message broker.

#### Method 2 - automated dealing with simple text file

1. Add new user `~/repositories/AI-CPS/code/messageBroker/users.txt` with password `newUserPassword` by

```
   mosquitto_passwd -b ~/repositories/AI-CPS/code/messageBroker/users.txt newUser newUserPassword
```

1. Delete user `~/repositories/AI-CPS/code/messageBroker/users.txt` from password file:

```
   mosquitto_passwd -D ~/repositories/AI-CPS/code/messageBroker/users.txt userToBeDeleted 
```

## Start message broker server with custom config file

```
   /usr/local/sbin/mosquitto -c ~/repositories/AI-CPS/code/messageBroker/mosquitto.conf
```

## Establish connection with user account

### Manually via CLI

For instance, publish a message from CLI with account `user1` having password `password1`,
which has been encrypted and loaded by message broker at `localhost` at port `1883`:

```
   mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=apply_knnSolution, knowledge_base=marcusgrum/knowledgebase_apple_banana_orange_pump_20, activation_base=marcusgrum/activationbase_apple_okay_01, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "localhost" -p 1883
```

Details can be found at the manual page [mosquitto_pub manual page](https://mosquitto.org/man/mosquitto_pub-1.html).

### Code-based via Python

For instance, publish a message from Python script with account `user1` having password `password1`,
which has been encrypted and loaded by message broker at `localhost` at port `1883`:

```
   # establish connection of client and server
   client.username_pw_set(username="user1", password="password1")
   client.connect(MQTT_Broker, 1883, 60)
```

Please note, the `username_pw_set()` needs to be called before `connect()`.