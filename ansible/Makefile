ping:
	ansible --key-file ~/.ssh/jetstream -u ubuntu --module-name ping -i inventory.ini dev

play:
	ansible-playbook --key-file ~/.ssh/jetstream -u ubuntu -i inventory.ini -l dev playbook.yaml

kill:
	ansible-playbook --key-file ~/.ssh/jetstream -u ubuntu -i inventory.ini -l dev kill.yaml


