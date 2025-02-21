ping:
	ansible --key-file ~/.ssh/jetstream -u exouser --module-name ping -i inventory.ini prod

play:
	ansible-playbook --key-file ~/.ssh/jetstream -u exouser -i inventory.ini -l prod playbook.yaml

kill:
	ansible-playbook --key-file ~/.ssh/jetstream -u exouser -i inventory.ini -l prod kill.yaml


ping-dev:
	ansible --key-file ~/.ssh/jetstream -u exouser --module-name ping -i inventory.ini dev

play-dev:
	ansible-playbook --key-file ~/.ssh/jetstream -u exouser -i inventory.ini -l dev playbook.yaml

kill-dev:
	ansible-playbook --key-file ~/.ssh/jetstream -u exouser -i inventory.ini -l dev kill.yaml
