ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
rm ~/.bashrc && ln -s bashrc ~/.bashrc