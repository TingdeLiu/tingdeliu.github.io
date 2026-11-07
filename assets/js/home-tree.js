const treeToggle = document.querySelector('[data-tree-toggle]');
const treeBranches = [...document.querySelectorAll('#home-branches details')];

function updateTreeToggle() {
  const expanded = treeBranches.every(branch => branch.open);
  treeToggle.textContent = expanded ? treeToggle.dataset.collapse : treeToggle.dataset.expand;
  treeToggle.setAttribute('aria-expanded', String(expanded));
}

if (treeToggle && treeBranches.length) {
  treeToggle.hidden = false;
  updateTreeToggle();
  treeToggle.addEventListener('click', () => {
    const expand = !treeBranches.every(branch => branch.open);
    treeBranches.forEach(branch => { branch.open = expand; });
    updateTreeToggle();
  });
  treeBranches.forEach(branch => branch.addEventListener('toggle', updateTreeToggle));
}
