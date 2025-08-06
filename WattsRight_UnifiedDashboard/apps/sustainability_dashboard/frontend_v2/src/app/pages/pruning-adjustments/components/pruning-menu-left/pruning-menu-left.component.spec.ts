import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PruningMenuLeftComponent } from './pruning-menu-left.component';

describe('PruningLeftMenuComponent', () => {
  let component: PruningMenuLeftComponent;
  let fixture: ComponentFixture<PruningMenuLeftComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PruningMenuLeftComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PruningMenuLeftComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
